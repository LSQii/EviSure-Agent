"""Cross-view stability gate for the completed post-answer verifier.

This stage never calls the Qwen API. It reuses the frozen verifier plans, audits and
consensus decisions, and runs two additional local LLaVA confirmations only for
the REOPEN cases. A changed answer is retained only when the stored verifier vote
and both new cross-view votes unanimously select the audited competitor.

Full run:
    python post_answer_stability_gate.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import causal_evidence_verifier as causal_verifier
import evidence_gap_reflective_agent as core
import post_answer_causal_verifier as post_verifier


HERE = Path(__file__).resolve().parent
DEFAULT_VERIFIER_RESULTS = HERE / "result" / "post_answer_causal_verifier" / "results.jsonl"
DEFAULT_SOURCE_RESULTS = HERE / "result" / "evidence_gap_reflective" / "results.jsonl"
DEFAULT_OUTPUT_DIR = HERE / "result" / "post_answer_stability_gate"

PROTOCOL_VERSION = "post-answer-three-vote-cross-view-stability"
EXPECTED_FULL_QUESTIONS = 1368
EXPECTED_TARGET_QUESTIONS = 151
EXPECTED_REOPEN_QUESTIONS = 17
GLOBAL_FRAMES = 16
WIDE_GLOBAL_FRAMES = 32
BASE_METHOD = "agent_pre_reflection"


def option(value) -> str:
    return core.extract_letter(value)


def method_result(prediction: str, raw: str, ground_truth: str) -> dict:
    return {
        "prediction": prediction,
        "raw_output": raw or prediction,
        "correct": prediction == ground_truth,
    }


def parse_global_timestamps(time_string: str) -> list[float]:
    values = []
    for value in str(time_string or "").split(","):
        value = value.strip().lower().removesuffix("s")
        if value:
            values.append(float(value))
    return values


def reconstruct_audit_frames(item: dict, record: dict) -> dict:
    requested = [float(value) for value in record.get("audit_frames", {}).get("timestamps", [])]
    raw, timestamps = core.sample_unique_frames(item["video_path"], requested, [])
    return {
        "frames": core.stamp_frames(raw, timestamps, "F"),
        "timestamps": timestamps,
        "frame_ids": [f"F{i + 1}" for i in range(len(timestamps))],
    }


def selected_audit_view(record: dict, sampled: dict):
    selected_ids = list(dict.fromkeys(record.get("consensus", {}).get("selected_evidence_ids", [])))
    route = record.get("verifier_route")
    limit = post_verifier.MAX_FINAL_ORDER_FRAMES if route == post_verifier.ROUTE_TEMPORAL_ORDER else post_verifier.MAX_FINAL_CAUSAL_FRAMES
    frames, timestamps, retained_ids = post_verifier.selected_audit_frames(sampled, selected_ids, limit)
    return frames, timestamps, retained_ids


def confirmation_prompt(source: dict, retained_ids: list[str], audit_ts: list[float], variant: int) -> str:
    mapping = ", ".join(f"{value}={timestamp:.2f}s" for value, timestamp in zip(retained_ids, audit_ts))
    if variant == 1:
        note = (
            "A blinded structured evidence audit selected additional timestamped raw frames "
            f"({mapping}). Independently reconstruct the visible event identities and their "
            "chronological order. Do not assume that the audit or any previous answer is correct."
        )
    else:
        note = (
            "Perform a skeptical independent re-check using the wider global video view and the "
            f"timestamped audit frames ({mapping}). Treat the selected windows as untrusted "
            "observations: check for earlier occurrences, confused objects, and option-order "
            "mismatches before selecting the answer. Do not follow any previous answer."
        )
    return core.insert_evidence(source["agent_prompt"], [note])


def run_confirmation(baseline, item: dict, source: dict, record: dict, sampled: dict, variant: int) -> tuple[str, str]:
    audit_frames, audit_ts, retained_ids = selected_audit_view(record, sampled)
    frame_count = GLOBAL_FRAMES if variant == 1 else WIDE_GLOBAL_FRAMES
    global_frames, time_string, _ = baseline.process_video(
        item["video_path"], frame_count, 1, force_sample=True,
    )
    global_ts = parse_global_timestamps(time_string)

    sources = [(global_frames, global_ts, "GLOBAL", 0)]
    if variant == 1:
        local_ts = causal_verifier.initial_local_timestamps(source)
        local_raw, local_ts = core.sample_unique_frames(item["video_path"], local_ts, global_ts)
        local_frames = core.stamp_frames(local_raw, local_ts, "L")
        sources.append((local_frames, local_ts, "LOCAL", 1))
    sources.append((audit_frames, audit_ts, "AUDIT", 2))
    merged, _, _ = causal_verifier.merge_sources(sources)
    prompt = confirmation_prompt(source, retained_ids, audit_ts, variant)
    raw = core.infer_frames(baseline, prompt, merged)
    return option(raw), raw


def confirmation_signature(item: dict, source: dict, record: dict) -> str:
    return core.stable_hash({
        "protocol": PROTOCOL_VERSION,
        "question_id": item["question_id"],
        "video_path": str(item["video_path"]),
        "source_fingerprint": source.get("config_fingerprint", ""),
        "verifier_fingerprint": record.get("config_fingerprint", ""),
        "competitor": record.get("consensus", {}).get("competitor_option", ""),
        "selected_evidence_ids": record.get("consensus", {}).get("selected_evidence_ids", []),
        "audit_timestamps": record.get("audit_frames", {}).get("timestamps", []),
        "global_frames": GLOBAL_FRAMES,
        "wide_global_frames": WIDE_GLOBAL_FRAMES,
    })


def make_output_record(verifier_record: dict, confirmation: dict | None, fingerprint: str) -> dict:
    ground_truth = verifier_record["ground_truth"]
    reviewed = option(verifier_record["methods"]["answer_before_causal"]["prediction"])
    verifier_candidate = option(verifier_record["methods"]["causal_candidate"]["prediction"])
    verifier_final = option(verifier_record["methods"]["answer_after_causal"]["prediction"])
    competitor = option(verifier_record.get("consensus", {}).get("competitor_option"))
    reopen = bool(verifier_record.get("consensus", {}).get("reopen_ready"))
    pass1 = option((confirmation or {}).get("pass1", {}).get("prediction"))
    pass2 = option((confirmation or {}).get("pass2", {}).get("prediction"))
    votes = [verifier_candidate, pass1, pass2]
    stable = bool(
        reopen and competitor and competitor != reviewed
        and all(value == competitor for value in votes)
    )
    final = competitor if stable else reviewed
    accepted = bool(stable and final != reviewed)
    if not reopen:
        reasons = list(verifier_record.get("consensus", {}).get("reasons", []))
    elif accepted:
        reasons = []
    else:
        reasons = [
            "The stored verifier vote and two cross-view local confirmations were not unanimous for the audited competitor."
        ]
    return {
        "question_id": verifier_record["question_id"],
        "question_level": verifier_record["question_level"],
        "task_type": verifier_record["task_type"],
        "question": verifier_record["question"],
        "ground_truth": ground_truth,
        "verifier_route": verifier_record.get("verifier_route"),
        "config_fingerprint": fingerprint,
        "methods": {
            "answer_before_stability": method_result(verifier_final, verifier_final, ground_truth),
            "verifier_local_vote": method_result(verifier_candidate, verifier_record["methods"]["causal_candidate"].get("raw_output", ""), ground_truth),
            "confirmation_pass1": method_result(pass1, (confirmation or {}).get("pass1", {}).get("raw_output", ""), ground_truth),
            "confirmation_pass2": method_result(pass2, (confirmation or {}).get("pass2", {}).get("raw_output", ""), ground_truth),
            "answer_after_stability": method_result(final, final, ground_truth),
        },
        "consensus": verifier_record.get("consensus", {}),
        "verifier_gate": verifier_record.get("gate", {}),
        "stability_gate": {
            "reopen_ready": reopen,
            "audited_competitor": competitor,
            "votes": votes,
            "unanimous": stable,
            "accepted_change": accepted,
            "decision": "STABLE_CHANGE_ACCEPTED" if accepted else "KEEP_REVIEWED",
            "prediction": final,
            "reasons": reasons,
        },
        "confirmation": confirmation or {},
    }


def summarize(records: dict[str, dict], source_records: dict[str, dict]) -> dict:
    values = list(records.values())
    source_values = list(source_records.values())
    source_base_correct = sum(
        value.get("methods", {}).get(BASE_METHOD, {}).get("correct", False)
        for value in source_values
    )
    paired_correct = sum(
        value.get("methods", {}).get("baseline_paired", {}).get("correct", False)
        for value in source_values
    )
    verifier_target_correct = sum(value["methods"]["answer_before_stability"]["correct"] for value in values)
    final_target_correct = sum(value["methods"]["answer_after_stability"]["correct"] for value in values)
    verifier_accepted = sum(value.get("verifier_gate", {}).get("accepted_change", False) for value in values)
    retained = sum(value["stability_gate"]["accepted_change"] for value in values)
    corrected = sum(
        not value["methods"]["answer_before_stability"]["correct"]
        and value["methods"]["answer_after_stability"]["correct"]
        for value in values
    )
    corrupted = sum(
        value["methods"]["answer_before_stability"]["correct"]
        and not value["methods"]["answer_after_stability"]["correct"]
        for value in values
    )
    # Exact full-dataset counts are filled by write_outputs from full_predictions.
    return {
        "protocol": PROTOCOL_VERSION,
        "target_questions": len(values),
        "reopen_questions": sum(value["stability_gate"]["reopen_ready"] for value in values),
        "verifier_accepted_changes": verifier_accepted,
        "retained_stable_changes": retained,
        "reverted_verifier_changes": verifier_accepted - retained,
        "target_verifier_correct": verifier_target_correct,
        "target_final_correct": final_target_correct,
        "target_total": len(values),
        "target_verifier_accuracy": round(verifier_target_correct / len(values), 4) if values else 0.0,
        "target_final_accuracy": round(final_target_correct / len(values), 4) if values else 0.0,
        "stability_delta_corrected": corrected,
        "stability_delta_corrupted": corrupted,
        "vote_patterns": dict(Counter(
            "/".join(value["stability_gate"]["votes"])
            for value in values if value["stability_gate"]["reopen_ready"]
        )),
        "source_base_correct": source_base_correct,
        "paired_baseline_correct": paired_correct,
    }


def write_full_predictions(output_dir: Path, source_records: dict[str, dict], records: dict[str, dict]) -> dict:
    path = output_dir / "full_predictions.jsonl"
    counts = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for qid in sorted(source_records):
            source = source_records[qid]
            target = records.get(qid)
            baseline = option(source.get("methods", {}).get("baseline_paired", {}).get("prediction"))
            base = option(source.get("methods", {}).get(BASE_METHOD, {}).get("prediction"))
            verifier_prediction = option(target["methods"]["answer_before_stability"]["prediction"]) if target else base
            final = option(target["methods"]["answer_after_stability"]["prediction"]) if target else base
            ground_truth = option(source.get("ground_truth"))
            row = {
                "question_id": qid,
                "question_level": source.get("question_level"),
                "task_type": source.get("task_type"),
                "ground_truth": ground_truth,
                "baseline_paired_prediction": baseline,
                "baseline_paired_correct": baseline == ground_truth,
                "agent_pre_reflection_prediction": base,
                "agent_pre_reflection_correct": base == ground_truth,
                "verifier_prediction": verifier_prediction,
                "verifier_correct": verifier_prediction == ground_truth,
                "final_prediction": final,
                "final_correct": final == ground_truth,
                "stability_change_retained": bool(target and target["stability_gate"]["accepted_change"]),
            }
            counts["total"] += 1
            counts["paired_correct"] += row["baseline_paired_correct"]
            counts["base_correct"] += row["agent_pre_reflection_correct"]
            counts["verifier_correct"] += row["verifier_correct"]
            counts["final_correct"] += row["final_correct"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dict(counts)


def write_outputs(output_dir: Path, records: dict[str, dict], source_records: dict[str, dict]) -> None:
    summary = summarize(records, source_records)
    full_counts = write_full_predictions(output_dir, source_records, records)
    summary["full_dataset"] = {
        **full_counts,
        "paired_accuracy": round(full_counts.get("paired_correct", 0) / full_counts.get("total", 1), 4),
        "base_accuracy": round(full_counts.get("base_correct", 0) / full_counts.get("total", 1), 4),
        "verifier_accuracy": round(full_counts.get("verifier_correct", 0) / full_counts.get("total", 1), 4),
        "final_accuracy": round(full_counts.get("final_correct", 0) / full_counts.get("total", 1), 4),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    fields = [
        "question_id", "task_type", "ground_truth", "verifier_prediction",
        "vote1", "vote2", "vote3", "competitor", "retained", "final_prediction",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid in sorted(records):
            value = records[qid]
            votes = value["stability_gate"]["votes"]
            writer.writerow({
                "question_id": qid,
                "task_type": value["task_type"],
                "ground_truth": value["ground_truth"],
                "verifier_prediction": value["methods"]["answer_before_stability"]["prediction"],
                "vote1": votes[0], "vote2": votes[1], "vote3": votes[2],
                "competitor": value["stability_gate"]["audited_competitor"],
                "retained": value["stability_gate"]["accepted_change"],
                "final_prediction": value["methods"]["answer_after_stability"]["prediction"],
            })


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-view local-answer stability gate; no Qwen API calls")
    parser.add_argument("--verifier_results", default=str(DEFAULT_VERIFIER_RESULTS))
    parser.add_argument("--source_results", default=str(DEFAULT_SOURCE_RESULTS))
    parser.add_argument("--data_dir", default=str(core.DEFAULT_DATA_DIR))
    parser.add_argument("--json", default=str(core.DEFAULT_ANNOTATION))
    parser.add_argument("--level_jsonl", default=str(core.DEFAULT_LEVELS))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--allow_partial_dataset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verifier_path = Path(args.verifier_results).resolve()
    source_path = Path(args.source_results).resolve()
    verifier_records = core.load_jsonl(verifier_path)
    source_records = core.load_jsonl(source_path)
    reopen_ids = sorted(
        qid for qid, value in verifier_records.items()
        if value.get("consensus", {}).get("reopen_ready")
    )
    question_map = {
        value["question_id"]: value for value in core.build_question_list(
            Path(args.json).resolve(), Path(args.data_dir).resolve(), Path(args.level_jsonl).resolve(),
        )
    }
    runnable_reopen_ids = [qid for qid in reopen_ids if qid in question_map]
    if not args.allow_partial_dataset and (
        len(source_records) != EXPECTED_FULL_QUESTIONS
        or len(verifier_records) != EXPECTED_TARGET_QUESTIONS
        or len(reopen_ids) != EXPECTED_REOPEN_QUESTIONS
        or len(runnable_reopen_ids) != EXPECTED_REOPEN_QUESTIONS
    ):
        raise RuntimeError(
            "Full-data guard failed: "
            f"source={len(source_records)}/{EXPECTED_FULL_QUESTIONS}, "
            f"verifier={len(verifier_records)}/{EXPECTED_TARGET_QUESTIONS}, "
            f"reopen={len(reopen_ids)}/{EXPECTED_REOPEN_QUESTIONS}, "
            f"runnable_reopen={len(runnable_reopen_ids)}/{EXPECTED_REOPEN_QUESTIONS}."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": PROTOCOL_VERSION,
        "verifier_results_hash": core.file_hash(verifier_path),
        "source_results_hash": core.file_hash(source_path),
        "reopen_ids_hash": core.stable_hash(reopen_ids),
        "global_frames": GLOBAL_FRAMES,
        "wide_global_frames": WIDE_GLOBAL_FRAMES,
        "local_confirmation_votes": 2,
        "stored_verifier_vote_required": True,
        "qwen_api_calls": 0,
    }
    fingerprint = core.stable_hash(config)
    manifest_path = output_dir / "manifest.json"
    cache_path = output_dir / "confirmation_cache.jsonl"
    results_path = output_dir / "results.jsonl"
    caches_exist = any(path.exists() and path.stat().st_size for path in (cache_path, results_path))
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("config_fingerprint") != fingerprint and caches_exist:
            raise RuntimeError("Output directory contains a different experiment; use a new --output_dir.")
    elif caches_exist:
        raise RuntimeError("Output directory contains unversioned caches; use a new --output_dir.")
    manifest = {
        "source_questions": len(source_records),
        "verifier_target_questions": len(verifier_records),
        "verifier_reopen_questions": len(reopen_ids),
        "runnable_reopen_questions": len(runnable_reopen_ids),
        "config_fingerprint": fingerprint,
        "config": config,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Manifest]", manifest)
    if args.prepare_only:
        return 0

    confirmations = core.load_jsonl(cache_path)
    records = core.load_jsonl(results_path)
    pending_reopens = [
        qid for qid in runnable_reopen_ids
        if confirmations.get(qid, {}).get("request_signature")
        != confirmation_signature(question_map[qid], source_records[qid], verifier_records[qid])
    ]
    baseline = None
    if pending_reopens:
        import video_evidence_backend as baseline

    for index, qid in enumerate(pending_reopens, 1):
        item, source, record = question_map[qid], source_records[qid], verifier_records[qid]
        print(f"[{index}/{len(pending_reopens)}] {qid} cross-view confirmations")
        sampled = reconstruct_audit_frames(item, record)
        pass1_prediction, pass1_raw = run_confirmation(baseline, item, source, record, sampled, 1)
        pass2_prediction, pass2_raw = run_confirmation(baseline, item, source, record, sampled, 2)
        confirmation = {
            "question_id": qid,
            "request_signature": confirmation_signature(item, source, record),
            "pass1": {"prediction": pass1_prediction, "raw_output": pass1_raw, "view": "standard_local"},
            "pass2": {"prediction": pass2_prediction, "raw_output": pass2_raw, "view": "wide_global"},
        }
        core.append_jsonl(cache_path, confirmation)
        confirmations[qid] = confirmation

    for qid, record in verifier_records.items():
        confirmation = confirmations.get(qid) if record.get("consensus", {}).get("reopen_ready") else None
        output = make_output_record(record, confirmation, fingerprint)
        old = records.get(qid)
        if old != output:
            core.append_jsonl(results_path, output)
            records[qid] = output

    write_outputs(output_dir, records, source_records)
    print(f"[Complete] confirmations={len(confirmations)}/{len(reopen_ids)}, targets={len(records)}/{len(verifier_records)}")
    print("[Results]", results_path)
    print("[Summary]", output_dir / "summary.json")
    print("[Full predictions]", output_dir / "full_predictions.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
