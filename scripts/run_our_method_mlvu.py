"""MLVU entry point for the final relationally verified Video QA method.

Only the dataset adapter differs from ``run_our_method_videomme.py``:

* every multiple-choice JSON file under ``--json_dir`` is one MLVU task;
* the official ``sub_scene`` and ``summary`` generation tasks are skipped;
* videos are resolved under ``--video_dir/<task>/``;
* the question and ``candidates`` are composed into one multiple-choice prompt;
* textual ground-truth answers are converted back to option letters.

The initial answer, evidence cache, routing, C-level relation verifier, stability
gate, resume behavior, and result format are shared with the other runners.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as defaults
from scripts import run_our_method_longvideobench as mc_compat
from scripts import run_our_method_videomme as runner


HERE = PROJECT_ROOT
OPTION_LETTERS = "ABCD"
ADAPTER_VERSION = "mlvu-baseline-compatible"
MLVU_GENERATION_TASKS = {"sub_scene", "summary"}


def path_from_here(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (HERE / path).resolve()


def discover_mlvu_tasks(json_dir: Path) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    skipped: list[str] = []
    for path in sorted(json_dir.glob("*.json")):
        # Official MLVU Dev names these files ``8_sub_scene.json`` and
        # ``9_summary.json``.  They are free-form generation tasks and have no
        # multiple-choice candidates, so they are outside this runner's A-D
        # answer/verifier protocol.  Also accept the same names without the
        # numeric prefix.
        normalized_task = re.sub(r"^\d+[_-]*", "", path.stem.casefold())
        if normalized_task in MLVU_GENERATION_TASKS:
            skipped.append(path.name)
            continue
        tasks.append((path.stem, path.resolve()))
    if skipped:
        print(
            "[Dataset] Skipping MLVU free-form generation tasks: "
            + ", ".join(skipped)
        )
    if not tasks:
        raise RuntimeError(
            "No supported MLVU multiple-choice JSON files found directly under: "
            f"{json_dir}"
        )
    return tasks


def read_mlvu_task(path: Path) -> list[dict]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        for key in ("data", "items", "annotations", "questions"):
            if isinstance(loaded.get(key), list):
                loaded = loaded[key]
                break
    if not isinstance(loaded, list) or any(not isinstance(row, dict) for row in loaded):
        raise ValueError(
            f"{path} must contain a JSON list of question objects, or an object "
            "containing data/items/annotations/questions."
        )
    return loaded


def load_level_map(path: Path | None) -> dict[str, dict]:
    result: dict[str, dict] = {}
    if path is None:
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            qid = str(
                value.get("question_id") or value.get("qid") or value.get("id") or ""
            ).strip()
            if not qid:
                raise ValueError(f"Level JSONL line {line_number} has no question ID.")
            level = str(
                value.get("question_level") or value.get("level") or "Unknown"
            ).upper()
            if level not in runner.VALID_LEVELS:
                level = "Unknown"
            result[qid] = {
                "question_level": level,
                "task_type": str(
                    value.get("task_type")
                    or value.get("route")
                    or value.get("route_name")
                    or "Unknown"
                ),
            }
    return result


def build_mlvu_video_map(
    video_dir: Path,
) -> tuple[dict[tuple[str, str], Path], dict[str, list[str]]]:
    """Map (task folder, exact filename) to a video without cross-task clashes."""

    video_map: dict[tuple[str, str], Path] = {}
    collisions: dict[str, list[str]] = {}
    for task_dir in sorted(video_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_key = task_dir.name.casefold()
        for path in sorted(task_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in runner.VIDEO_SUFFIXES:
                continue
            resolved = path.resolve()
            keys = {
                (task_key, path.name.casefold()),
                (task_key, path.stem.casefold()),
            }
            for key in keys:
                if key in video_map and video_map[key] != resolved:
                    label = f"{key[0]}::{key[1]}"
                    collisions.setdefault(label, [str(video_map[key])]).append(str(resolved))
                else:
                    video_map[key] = resolved
    return video_map, collisions


def resolve_mlvu_video_path(
    task: str,
    record: dict,
    video_dir: Path,
    video_map: dict[tuple[str, str], Path],
) -> Path | None:
    raw = str(
        record.get("video")
        or record.get("video_path")
        or record.get("video_file")
        or record.get("file_name")
        or ""
    ).strip()
    if not raw:
        return None

    supplied = Path(raw)
    if supplied.is_absolute() and supplied.is_file():
        return supplied.resolve()
    for candidate in (video_dir / task / supplied, video_dir / supplied):
        if candidate.is_file():
            return candidate.resolve()

    task_key = task.casefold()
    for name in (supplied.name.casefold(), supplied.stem.casefold()):
        matched = video_map.get((task_key, name))
        if matched:
            return matched
    return None


def normalise_mlvu_choices(
    record: dict,
) -> tuple[list[str], list[str], list[str], list[int]]:
    choices = record.get("candidates")
    if choices is None:
        choices = record.get("choices", record.get("options", []))
    if not isinstance(choices, list):
        raise ValueError("MLVU candidates/choices must be a list.")
    # Match the supplied MLVU baseline exactly: candidate positions are
    # semantic, even when an annotation contains an empty string.  In the
    # official 3_ego record 120, both the third candidate and ``answer`` are
    # empty; the baseline consequently maps that ground truth to option C.
    # Compacting the list as Video-MME does would change the option letters and
    # make this MLVU record impossible to score consistently with the baseline.
    raw_values = [str(value).strip() for value in choices]
    retained_indices = list(range(len(raw_values)))
    values = raw_values
    if not 2 <= len(values) <= len(OPTION_LETTERS):
        raise ValueError(
            "This runner requires 2-4 MLVU candidate positions because the "
            f"method's relation verifier uses the A-D protocol; received {len(values)}."
        )
    labels = list(OPTION_LETTERS[: len(values)])
    labelled = [
        value
        if re.match(r"^\s*[A-D][\.\)\:]\s*", value, flags=re.I)
        else f"{label}. {value}"
        for label, value in zip(labels, values)
    ]
    return values, labelled, labels, retained_indices


def compose_mlvu_question(question: Any, labelled_choices: list[str]) -> str:
    text = str(question or "").strip()
    if not text:
        raise ValueError("MLVU question text is empty.")
    return "\n".join([text, *labelled_choices])


def convert_mlvu_answer(
    value: Any,
    choices: list[str],
    labels: list[str],
    retained_indices: list[int],
) -> str:
    text = str(value if value is not None else "").strip()
    for index, candidate in enumerate(choices):
        if text.casefold() == candidate.casefold():
            return labels[index]
    upper = text.upper()
    match = re.search(r"\b([A-Z])\b", upper)
    if match:
        original_index = ord(match.group(1)) - ord("A")
        if original_index in retained_indices:
            return labels[retained_indices.index(original_index)]
    return ""


def make_mlvu_item(
    task: str,
    record: dict,
    index: int,
    video_dir: Path,
    video_map: dict[tuple[str, str], Path],
    level_map: dict[str, dict],
) -> dict:
    raw_qid = str(
        record.get("id")
        or record.get("question_id")
        or record.get("qid")
        or f"{index:05d}"
    ).strip()
    # MLVU IDs may repeat across task JSONs; the composite ID is resume-safe.
    qid = f"{task}:{raw_qid}"
    (
        choices,
        labelled_choices,
        option_letters,
        retained_choice_indices,
    ) = normalise_mlvu_choices(record)
    video_path = resolve_mlvu_video_path(task, record, video_dir, video_map)
    video_name = str(
        record.get("video")
        or record.get("video_path")
        or record.get("video_file")
        or record.get("file_name")
        or ""
    ).strip()

    answer_present = any(
        key in record for key in ("answer", "ground_truth", "label", "correct_choice")
    )
    raw_answer = record.get(
        "answer",
        record.get(
            "ground_truth",
            record.get("label", record.get("correct_choice", "")),
        ),
    )
    ground_truth = convert_mlvu_answer(
        raw_answer,
        choices,
        option_letters,
        retained_choice_indices,
    )
    if answer_present and not ground_truth:
        raise ValueError(
            f"MLVU item {qid} has invalid answer {raw_answer!r} for candidates {choices!r}."
        )

    external = level_map.get(qid, level_map.get(raw_qid, {}))
    level = str(
        record.get("question_level")
        or record.get("level")
        or external.get("question_level")
        or "Unknown"
    ).upper()
    task_type = str(
        record.get("task_type")
        or external.get("task_type")
        or "Unknown"
    )
    question_type = str(
        record.get("question_type") or record.get("type") or task
    )
    return {
        "question_id": qid,
        "source_question_id": raw_qid,
        "video_id": f"{task}-{Path(video_name).name}" if video_name else task,
        "video_path": str(video_path) if video_path else "",
        "question": compose_mlvu_question(record.get("question"), labelled_choices),
        "choices": labelled_choices,
        "option_letters": option_letters,
        "ground_truth": ground_truth,
        "question_level": level,
        "task_type": task_type,
        "task": task,
        "mlvu_task": task,
        "question_type": question_type,
        "video_name": video_name,
        "video_category": "MLVU",
        "video_subcategory": task,
        "duration": record.get("duration"),
    }


def load_mlvu_dataset(
    json_dir: Path,
    video_dir: Path,
    level_jsonl: Path | None,
) -> tuple[list[dict], list[dict], dict[str, list[str]], list[Path]]:
    tasks = discover_mlvu_tasks(json_dir)
    level_map = load_level_map(level_jsonl)
    video_map, collisions = build_mlvu_video_map(video_dir)
    items: list[dict] = []
    missing: list[dict] = []
    invalid: list[str] = []
    annotation_files: list[Path] = []
    for task, json_path in tasks:
        annotation_files.append(json_path)
        for index, record in enumerate(read_mlvu_task(json_path), 1):
            try:
                item = make_mlvu_item(
                    task, record, index, video_dir, video_map, level_map
                )
            except ValueError as exc:
                video_name = str(
                    record.get("video")
                    or record.get("video_path")
                    or record.get("video_file")
                    or record.get("file_name")
                    or ""
                ).strip()
                invalid.append(
                    f"Invalid MLVU item {json_path.name} record {index} "
                    f"(video={video_name!r}): {exc}"
                )
                continue
            (items if item["video_path"] else missing).append(item)
    if invalid:
        print(
            f"[Dataset] Skipped {len(invalid)} malformed MLVU records, matching "
            "the supplied baseline's per-item skip behavior."
        )
        for message in invalid[:8]:
            print(f"  [Skip] {message}")
        if len(invalid) > 8:
            print(f"  [Skip] ... and {len(invalid) - 8} more malformed records")
    duplicates = [
        qid
        for qid, count in Counter(item["question_id"] for item in [*items, *missing]).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate MLVU question IDs: {duplicates[:5]}")
    return items, missing, collisions, annotation_files


def annotations_hash(files: list[Path]) -> str:
    return runner.stable_hash(
        [{"name": path.name, "sha256": runner.file_hash(path)} for path in files]
    )


def manifest_config(
    args: argparse.Namespace,
    annotation_files: list[Path],
    items: list[dict],
    frame_budget: int,
    wide_frames: int,
) -> dict:
    return {
        "protocol": runner.PROTOCOL,
        "dataset_adapter": ADAPTER_VERSION,
        "json_dir": str(args.json_dir),
        "annotation_files": [path.name for path in annotation_files],
        "annotation_hash": annotations_hash(annotation_files),
        "question_ids_hash": runner.stable_hash(
            [item["question_id"] for item in items]
        ),
        "dataset_format": args.dataset_format,
        "answer_frames": frame_budget,
        "wide_global_frames": wide_frames,
        "planner_frames": args.planner_frames,
        "relation_storyboard_frames": args.relation_storyboard_frames,
        "routing_mode": args.routing_mode,
        "api_base": args.api_base,
        "api_model": args.api_model,
        "save_diagnostics": args.save_diagnostics,
        "level_jsonl": str(args.level_jsonl) if args.level_jsonl else "",
        "level_jsonl_hash": (
            runner.file_hash(args.level_jsonl) if args.level_jsonl else ""
        ),
        "option_letters": OPTION_LETTERS,
    }


def summarise_group(records: list[dict], field: str) -> dict[str, dict]:
    buckets: dict[str, dict] = {}
    for row in records:
        key = str(row.get(field) or "Unknown")
        bucket = buckets.setdefault(
            key, {"completed": 0, "scored": 0, "correct": 0}
        )
        bucket["completed"] += 1
        if row.get("ground_truth"):
            bucket["scored"] += 1
            bucket["correct"] += int(bool(row.get("correct")))
    for bucket in buckets.values():
        scored = bucket["scored"]
        bucket["accuracy"] = (
            round(bucket["correct"] / scored, 4) if scored else None
        )
    return dict(sorted(buckets.items()))


def write_mlvu_outputs(
    output_dir: Path,
    records: dict[str, dict],
    frame_budget: int,
    wide_frames: int,
) -> None:
    """Write common method outputs plus MLVU task/question-type views."""

    runner.write_outputs(output_dir, records, frame_budget, wide_frames)
    values = [records[key] for key in sorted(records)]
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["by_mlvu_task"] = summarise_group(values, "mlvu_task")
    summary["by_question_type"] = summarise_group(values, "question_type")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = [
        "mlvu_task",
        "source_question_id",
        "question_id",
        "question_type",
        "video_id",
        "video_path",
        "question_level",
        "task_type",
        "route_source",
        "ground_truth",
        "initial_prediction",
        "final_prediction",
        "correct",
        "status",
        "relation_target",
        "verifier_route",
        "accepted_change",
        "elapsed_seconds",
        "error",
    ]
    with (output_dir / "predictions_mlvu.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_budget(
    args: argparse.Namespace,
    items: list[dict],
    annotation_files: list[Path],
    frame_budget: int,
) -> None:
    wide_frames = args.wide_frames or frame_budget * 2
    output_dir = args.output_dir / f"frames_{frame_budget}"
    config = manifest_config(
        args, annotation_files, items, frame_budget, wide_frames
    )
    fingerprint = runner.ensure_manifest(output_dir, config)
    records = runner.load_jsonl_by_id(output_dir / "predictions.jsonl")
    pending = [
        item
        for item in items
        if records.get(item["question_id"], {}).get("config_fingerprint")
        != fingerprint
        or (
            args.retry_errors
            and records.get(item["question_id"], {}).get("status")
            in {"error", "verifier_fallback"}
        )
    ]
    print(
        f"[frames={frame_budget}] completed={len(records)}/{len(items)}, "
        f"pending={len(pending)}"
    )
    if args.prepare_only:
        write_mlvu_outputs(output_dir, records, frame_budget, wide_frames)
        return

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    os.chdir(HERE)
    import video_evidence_backend as baseline
    import evidence_gap_reflective_agent as core
    import causal_evidence_verifier as causal_verifier
    import post_answer_causal_verifier as post_verifier
    import post_answer_stability_gate as stability_gate

    mc_compat.install_longvideobench_compat(baseline, core, post_verifier)
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
    cache_dir = output_dir / "evidence_cache"
    prediction_path = output_dir / "predictions.jsonl"
    for index, raw_item in enumerate(pending, 1):
        print(f"\n[{index}/{len(pending)}] {raw_item['question_id']}")
        mc_compat.ACTIVE_OPTION_LETTERS = "".join(
            raw_item.get("option_letters") or OPTION_LETTERS
        )
        item = runner.route_item(core, raw_item, args)
        source: dict = {}
        initial_diag: dict = {}
        relation = None
        stability = None
        status = "error"
        error = ""
        started = time.perf_counter()
        try:
            source, initial_diag = runner.run_initial_answer(
                baseline, core, item, args, cache_dir
            )
            target = (
                item["question_level"] == "C"
                and item["task_type"] in runner.ROUTE_TASKS
            )
            if not target:
                status = "pass_through"
            else:
                try:
                    relation = runner.run_relation_verifier(
                        baseline, core, causal_verifier, post_verifier, item, source, args
                    )
                    if relation.get("consensus", {}).get("reopen_ready"):
                        stability = runner.run_stability_gate(
                            baseline, stability_gate, item, source, relation, fingerprint
                        )
                        status = (
                            "stable_change"
                            if stability["stability_gate"]["accepted_change"]
                            else "unstable_keep"
                        )
                    else:
                        status = "verified_keep"
                except Exception as verifier_exc:
                    status = "verifier_fallback"
                    error = f"{type(verifier_exc).__name__}: {verifier_exc}"
                    print(f"[Verifier fallback] {item['question_id']}: {error}")
            record = runner.compact_record(
                item,
                source,
                initial_diag,
                relation,
                stability,
                fingerprint,
                status,
                error,
                args.save_diagnostics,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[Error] {item['question_id']}: {error}")
            record = {
                **item,
                "config_fingerprint": fingerprint,
                "status": "error",
                "error": error,
                "initial_prediction": "",
                "final_prediction": "",
                "correct": None,
                "relation_target": False,
                "verifier_route": "NOT_RUN",
                "accepted_change": False,
            }
        record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        runner.append_jsonl(prediction_path, record)
        records[item["question_id"]] = record
        write_mlvu_outputs(output_dir, records, frame_budget, wide_frames)
        print(
            "[Done]",
            item["question_id"],
            "level=",
            item["question_level"],
            "task=",
            item["task_type"],
            "initial=",
            record.get("initial_prediction"),
            "final=",
            record.get("final_prediction"),
            "status=",
            record.get("status"),
        )
    write_mlvu_outputs(output_dir, records, frame_budget, wide_frames)
    print(f"[Complete] {len(records)}/{len(items)} -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final Video QA method on MLVU."
    )
    parser.add_argument(
        "--json_dir",
        required=True,
        help="Directory containing one MLVU JSON file per task.",
    )
    parser.add_argument(
        "--video_dir",
        required=True,
        help="MLVU video root containing task-named subdirectories.",
    )
    parser.add_argument("--output_dir", default="outputs/mlvu")
    parser.add_argument(
        "--level_jsonl",
        default=None,
        help="Optional JSONL mapping question IDs to question_level/task_type.",
    )
    parser.add_argument(
        "--answer_frames",
        nargs="+",
        type=int,
        choices=runner.SUPPORTED_FRAME_BUDGETS,
        default=defaults.ANSWER_FRAMES,
    )
    parser.add_argument("--planner_frames", type=int, default=defaults.PLANNER_FRAMES)
    parser.add_argument(
        "--relation_storyboard_frames",
        type=int,
        default=defaults.RELATION_STORYBOARD_FRAMES,
    )
    parser.add_argument(
        "--wide_frames",
        type=int,
        default=None,
        help="Default: 2 x answer_frames.",
    )
    parser.add_argument(
        "--routing_mode",
        choices=("auto", "metadata"),
        default=defaults.ROUTING_MODE,
    )
    parser.add_argument(
        "--api_key",
        default=(
            os.environ.get("UNIFIED_QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or defaults.API_KEY
        ),
    )
    parser.add_argument("--api_base", default=defaults.API_BASE)
    parser.add_argument("--api_model", default=defaults.API_MODEL)
    parser.add_argument("--max_questions", type=int, default=defaults.MAX_QUESTIONS)
    parser.add_argument(
        "--allow_missing_videos",
        action="store_true",
        default=defaults.ALLOW_MISSING_VIDEOS,
    )
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--retry_errors", action="store_true")
    parser.add_argument(
        "--save_diagnostics",
        action="store_true",
        default=defaults.SAVE_DIAGNOSTICS,
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Do not save large nested planner/audit diagnostics.",
    )
    args = parser.parse_args()
    if args.compact:
        args.save_diagnostics = False
    args.json_dir = path_from_here(args.json_dir)
    args.video_dir = path_from_here(args.video_dir)
    args.output_dir = path_from_here(args.output_dir)
    args.level_jsonl = (
        path_from_here(args.level_jsonl) if args.level_jsonl else None
    )
    args.dataset_format = "mlvu"
    return args


def main() -> int:
    args = parse_args()
    if not args.json_dir.is_dir():
        raise NotADirectoryError(f"MLVU JSON directory not found: {args.json_dir}")
    if not args.video_dir.is_dir():
        raise NotADirectoryError(f"MLVU video directory not found: {args.video_dir}")
    if args.level_jsonl is not None and not args.level_jsonl.is_file():
        raise FileNotFoundError(f"Level JSONL not found: {args.level_jsonl}")
    if not args.prepare_only and not args.api_key:
        raise RuntimeError(
            "Qwen API key is empty. Set DASHSCOPE_API_KEY/UNIFIED_QWEN_API_KEY "
            "or fill API_KEY in config.py."
        )

    items, missing, collisions, annotation_files = load_mlvu_dataset(
        args.json_dir, args.video_dir, args.level_jsonl
    )
    if collisions:
        preview = ", ".join(sorted(collisions)[:8])
        print(
            f"[Warning] {len(collisions)} duplicate MLVU video keys found; "
            f"the first sorted path is used. Examples: {preview}"
        )
    if args.max_questions is not None:
        items = items[: max(0, args.max_questions)]
    if missing:
        preview = ", ".join(item["question_id"] for item in missing[:8])
        message = (
            f"{len(missing)} MLVU questions have no matching video; examples: {preview}"
        )
        if not args.allow_missing_videos:
            raise RuntimeError(
                message
                + ". Check that video subfolder names match JSON filenames, "
                "or use --allow_missing_videos."
            )
        print("[Warning] " + message)
    if not items:
        raise RuntimeError("No runnable MLVU questions remain after video matching.")

    option_counts = Counter(len(item["option_letters"]) for item in items)
    task_counts = Counter(item["mlvu_task"] for item in items)
    print(
        f"[MLVU] runnable={len(items)}, missing={len(missing)}, "
        f"tasks={dict(task_counts)}, option_counts={dict(option_counts)}, "
        f"frame_budgets={args.answer_frames}"
    )
    for frame_budget in args.answer_frames:
        run_budget(args, items, annotation_files, frame_budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
