"""LongVideoBench entry point for the final relationally verified Video QA method.

This runner reuses ``run_our_method_videomme.py`` and keeps its Video-MME data
path unchanged.
It converts LongVideoBench records into the canonical item structure expected by
the final method, then installs process-local A-E compatibility before models
are imported and inference starts.
"""

from __future__ import annotations

import argparse
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
from scripts import run_our_method_videomme as runner


HERE = PROJECT_ROOT
OPTION_LETTERS = "ABCDE"
ADAPTER_VERSION = "longvideobench-compatible"
ACTIVE_OPTION_LETTERS = OPTION_LETTERS


def read_longvideobench_annotation(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            loaded = [json.loads(line) for line in handle if line.strip()]
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            for key in ("data", "items", "annotations", "questions"):
                if isinstance(loaded.get(key), list):
                    loaded = loaded[key]
                    break
    if not isinstance(loaded, list) or any(not isinstance(value, dict) for value in loaded):
        raise ValueError(
            "LongVideoBench annotation must be a JSON/JSONL list of objects or "
            "an object containing data/items/annotations/questions."
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
            qid = str(value.get("question_id") or value.get("qid") or value.get("id") or "").strip()
            if not qid:
                raise ValueError(f"Level JSONL line {line_number} has no question ID.")
            level = str(value.get("question_level") or value.get("level") or "Unknown").upper()
            if level not in runner.VALID_LEVELS:
                level = "Unknown"
            result[qid] = {
                "question_level": level,
                "task_type": str(
                    value.get("task_type")
                    or value.get("question_type")
                    or value.get("route")
                    or value.get("route_name")
                    or "Unknown"
                ),
            }
    return result


def build_longvideobench_video_map(video_dir: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    video_map: dict[str, Path] = {}
    collisions: dict[str, list[str]] = {}
    for path in sorted(video_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in runner.VIDEO_SUFFIXES:
            continue
        resolved = path.resolve()
        if path.stem in video_map and video_map[path.stem] != resolved:
            collisions.setdefault(path.stem, [str(video_map[path.stem])]).append(str(resolved))
        else:
            video_map[path.stem] = resolved
    return video_map, collisions


def resolve_longvideobench_video_path(
    record: dict,
    video_dir: Path,
    video_map: dict[str, Path],
) -> Path | None:
    raw_path = str(
        record.get("video_path")
        or record.get("video")
        or record.get("video_file")
        or record.get("filepath")
        or record.get("file_name")
        or ""
    ).strip()
    if raw_path:
        supplied = Path(raw_path)
        if supplied.is_absolute() and supplied.is_file():
            return supplied.resolve()
        candidate = video_dir / supplied
        if candidate.is_file():
            return candidate.resolve()

    video_id = str(record.get("video_id") or "").strip()
    for key in (video_id, Path(video_id).stem if video_id else ""):
        if key and key in video_map:
            return video_map[key]
    return None


def normalise_longvideobench_choices(record: dict) -> tuple[list[str], list[str]]:
    choices = record.get("candidates")
    if choices is None:
        choices = record.get("choices", record.get("options", []))
    if not isinstance(choices, list):
        raise ValueError("LongVideoBench candidates/choices must be a list.")
    values = [str(value).strip() for value in choices]
    if any(not value for value in values):
        raise ValueError("LongVideoBench candidates cannot contain empty values.")
    if not 2 <= len(values) <= len(OPTION_LETTERS):
        raise ValueError("Each LongVideoBench question must have 2-5 candidates.")
    labels = list(OPTION_LETTERS[:len(values)])
    labelled = [
        value if re.match(r"^\s*[A-E][\.\)\:]\s*", value, flags=re.I) else f"{label}. {value}"
        for label, value in zip(labels, values)
    ]
    return labelled, labels


def compose_longvideobench_question(question: Any, labelled_choices: list[str]) -> str:
    text = str(question or "").strip()
    if not text:
        raise ValueError("LongVideoBench question text is empty.")
    return "\n".join([text, *labelled_choices])


def convert_longvideobench_answer(value: Any, option_letters: list[str]) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return option_letters[value] if 0 <= value < len(option_letters) else ""
    text = str(value if value is not None else "").strip().upper()
    if text.isdigit():
        index = int(text)
        return option_letters[index] if 0 <= index < len(option_letters) else ""
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match and match.group(1) in option_letters else ""


def make_longvideobench_item(
    record: dict,
    index: int,
    video_dir: Path,
    video_map: dict[str, Path],
    level_map: dict[str, dict],
) -> dict:
    qid = str(
        record.get("id")
        or record.get("question_id")
        or record.get("qid")
        or f"longvideobench-{index:05d}"
    ).strip()
    labelled_choices, option_letters = normalise_longvideobench_choices(record)
    video_path = resolve_longvideobench_video_path(record, video_dir, video_map)
    video_id = str(
        record.get("video_id")
        or Path(str(record.get("video_path") or "")).stem
        or "unknown"
    ).strip()

    answer_is_present = "correct_choice" in record or any(
        key in record for key in ("answer", "ground_truth", "label")
    )
    if "correct_choice" in record:
        raw_answer = record["correct_choice"]
    else:
        raw_answer = record.get("answer", record.get("ground_truth", record.get("label", "")))
    ground_truth = convert_longvideobench_answer(raw_answer, option_letters)
    if answer_is_present and not ground_truth:
        raise ValueError(
            f"LongVideoBench item {qid} has an invalid answer {raw_answer!r} "
            f"for options {option_letters}."
        )

    external_level = level_map.get(qid, {})
    level = str(
        record.get("question_level")
        or record.get("level")
        or external_level.get("question_level")
        or "Unknown"
    ).upper()
    task_type = str(
        record.get("task_type")
        or record.get("question_type")
        or record.get("type")
        or external_level.get("task_type")
        or "Unknown"
    )
    return {
        "question_id": qid,
        "video_id": video_id,
        "video_path": str(video_path) if video_path else "",
        "question": compose_longvideobench_question(record.get("question"), labelled_choices),
        "choices": labelled_choices,
        "option_letters": option_letters,
        "ground_truth": ground_truth,
        "question_level": level,
        "task_type": task_type,
        "video_category": str(
            record.get("video_category") or record.get("category") or "LongVideoBench"
        ),
        "video_subcategory": str(
            record.get("video_subcategory") or record.get("subcategory") or "Unknown"
        ),
        "duration": record.get("duration"),
    }


def load_longvideobench_dataset(
    annotation: Path,
    video_dir: Path,
    level_jsonl: Path | None,
) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    raw = read_longvideobench_annotation(annotation)
    level_map = load_level_map(level_jsonl)
    video_map, collisions = build_longvideobench_video_map(video_dir)
    items: list[dict] = []
    missing: list[dict] = []
    all_qids: list[str] = []
    for index, record in enumerate(raw, 1):
        item = make_longvideobench_item(record, index, video_dir, video_map, level_map)
        all_qids.append(item["question_id"])
        (items if item["video_path"] else missing).append(item)
    duplicates = [qid for qid, count in Counter(all_qids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate LongVideoBench question IDs: {duplicates[:5]}")
    return items, missing, collisions


def extract_longvideobench_letter(value: Any) -> str:
    text = str(value or "").strip().upper()
    allowed = re.escape(ACTIVE_OPTION_LETTERS)
    for pattern in (
        rf"(?:OPTION|ANSWER|CHOICE)[\s_:\-]*([{allowed}])\b",
        rf"\b([{allowed}])[\.\)](?:\s|$)",
        rf"\b([{allowed}])\b",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def question_option_letters(question: str) -> list[str]:
    matches = re.findall(r"(?:^|\n|\s)([A-E])\.\s+", str(question), flags=re.I)
    return list(dict.fromkeys(value.upper() for value in matches))


def install_longvideobench_compat(baseline: Any, core: Any, post_verifier: Any) -> None:
    """Install A-E support only inside this LongVideoBench process."""

    core.extract_letter = extract_longvideobench_letter
    baseline.extract_answer = extract_longvideobench_letter

    original_planner_prompt = core.planner_prompt

    def planner_prompt_ae(*args, **kwargs):
        prompt = original_planner_prompt(*args, **kwargs)
        question = str(args[0] if args else kwargs.get("question", ""))
        labels = question_option_letters(question) or list(ACTIVE_OPTION_LETTERS)
        option_text = "/".join(labels)
        return prompt.replace("A/B/C/D/UNKNOWN", f"{option_text}/UNKNOWN").replace(
            "A/B/C/D", option_text
        )

    core.planner_prompt = planner_prompt_ae

    def deterministic_option_orders_ae(question: str) -> dict[str, list[str]]:
        matches = re.findall(
            r"(?:^|\n|\s)([A-E])\.\s*(.*?)(?=(?:\n|\s)[A-E]\.\s|$)",
            question,
            flags=re.S | re.I,
        )
        if not 2 <= len(matches) <= 5:
            return {}
        circled = {
            "①": "E1", "②": "E2", "③": "E3",
            "④": "E4", "⑤": "E5", "⑥": "E6",
        }
        result: dict[str, list[str]] = {}
        for answer, text in matches:
            parenthetical = re.findall(r"\(([a-fA-F])\)", text)
            if parenthetical:
                sequence = [f"E{ord(value.lower()) - ord('a') + 1}" for value in parenthetical]
            else:
                sequence = [circled[value] for value in text if value in circled]
            if len(sequence) < 2 or len(sequence) != len(set(sequence)):
                return {}
            result[answer.upper()] = sequence
        expected_labels = question_option_letters(question)
        expected_events = set(result.get(expected_labels[0], [])) if expected_labels else set()
        if set(result) != set(expected_labels):
            return {}
        if any(set(value) != expected_events for value in result.values()):
            return {}
        return result

    def valid_order_plan_json_ae(parsed: dict, frame_count: int, question: str = "") -> bool:
        if not isinstance(parsed, dict):
            return False
        events = parsed.get("events")
        if not isinstance(events, list) or not 2 <= len(events) <= 6:
            return False
        expected_ids = [f"E{i + 1}" for i in range(len(events))]
        actual_ids = [
            str(value.get("event_id") or "").upper()
            for value in events if isinstance(value, dict)
        ]
        if actual_ids != expected_ids:
            return False
        for value in events:
            if not isinstance(value, dict) or not str(value.get("description") or "").strip():
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
        option_labels = question_option_letters(question)
        orders = parsed.get("option_orders")
        if not option_labels or not isinstance(orders, dict) or set(orders) != set(option_labels):
            return False
        for answer in option_labels:
            sequence = orders.get(answer)
            if (
                not isinstance(sequence, list)
                or len(sequence) != len(expected_ids)
                or set(sequence) != set(expected_ids)
            ):
                return False
        deterministic = deterministic_option_orders_ae(question)
        if deterministic and any(orders[answer] != deterministic[answer] for answer in option_labels):
            return False
        return True

    def order_planner_prompt_ae(question: str, timestamps: list[float], retry: bool = False) -> str:
        labels = question_option_letters(question)
        label_text = ", ".join(labels)
        storyboard = ", ".join(f"S{i}={value:.2f}s" for i, value in enumerate(timestamps))
        prefix = "Your previous JSON was invalid. " if retry else ""
        example_orders = ",".join(
            f'"{label}":["E1","E2","E3"]' for label in labels
        )
        return f"""{prefix}Build a COMPLETE event timeline for this multiple-choice video
order question. Extract every atomic event/tool/concept whose order is permuted by the
answer options. Assign E1..EN in the order listed in the question, not video order.
Locate the first relevant occurrence of every event in the storyboard. Transcribe every
option permutation using only event IDs.
Question and options: {question}
Storyboard labels: {storyboard}
Option labels: {label_text}
Rules: include every ordered item exactly once; 2<=N<=6; every event needs a nonempty
anchor_indices list; option_orders must contain exactly {label_text}, and each must be a
complete permutation of E1..EN; descriptions <=40 characters. Return one-line minified
JSON only, no markdown.
Schema: {{"events":[{{"event_id":"E1","description":"event","anchor_indices":[0]}},
{{"event_id":"E2","description":"event","anchor_indices":[1]}},
{{"event_id":"E3","description":"event","anchor_indices":[2]}}],
"option_orders":{{{example_orders}}}}}"""

    post_verifier.deterministic_option_orders = deterministic_option_orders_ae
    post_verifier.valid_order_plan_json = valid_order_plan_json_ae
    post_verifier.order_planner_prompt = order_planner_prompt_ae


def manifest_config(
    args: argparse.Namespace,
    annotation: Path,
    items: list[dict],
    frame_budget: int,
    wide_frames: int,
) -> dict:
    config = runner.manifest_config(args, annotation, items, frame_budget, wide_frames)
    config["dataset_adapter"] = ADAPTER_VERSION
    config["level_jsonl"] = str(args.level_jsonl) if args.level_jsonl else ""
    config["level_jsonl_hash"] = runner.file_hash(args.level_jsonl) if args.level_jsonl else ""
    config["option_letters"] = OPTION_LETTERS
    return config


def run_budget(
    args: argparse.Namespace,
    items: list[dict],
    annotation: Path,
    frame_budget: int,
) -> None:
    global ACTIVE_OPTION_LETTERS
    wide_frames = args.wide_frames or frame_budget * 2
    output_dir = runner.profiling_output_dir(args, frame_budget)
    config = manifest_config(args, annotation, items, frame_budget, wide_frames)
    fingerprint = runner.ensure_manifest(output_dir, config)
    records = runner.load_jsonl_by_id(output_dir / "predictions.jsonl")
    runner.restore_profile_seen_videos(args, output_dir)
    pending = [
        item for item in items
        if records.get(item["question_id"], {}).get("config_fingerprint") != fingerprint
        or (
            args.retry_errors
            and records.get(item["question_id"], {}).get("status")
            in {"error", "verifier_fallback"}
        )
    ]
    print(f"[frames={frame_budget}] completed={len(records)}/{len(items)}, pending={len(pending)}")
    if args.prepare_only:
        runner.write_outputs(output_dir, records, frame_budget, wide_frames)
        return

    cache_dir = runner.configure_profile_cache(args, frame_budget, output_dir)
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

    install_longvideobench_compat(baseline, core, post_verifier)
    if getattr(args, "profile", False):
        runner.complexity.install_runtime_wrappers(baseline, core, args)
        runner.complexity.write_run_manifest(
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
        ACTIVE_OPTION_LETTERS = "".join(raw_item.get("option_letters") or OPTION_LETTERS)
        runner.start_question_profile(args, raw_item)
        with runner.profile_stage(args, "routing"):
            item = runner.route_item(core, raw_item, args)
        source: dict = {}
        initial_diag: dict = {}
        relation = None
        stability = None
        status = "error"
        error = ""
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
                            baseline, stability_gate, item, source, relation, fingerprint, args
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
        runner.finish_question_profile(
            args, output_dir, record, initial_diag, relation, stability
        )
        runner.append_jsonl(prediction_path, record)
        records[item["question_id"]] = record
        runner.write_outputs(output_dir, records, frame_budget, wide_frames)
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
    runner.write_outputs(output_dir, records, frame_budget, wide_frames)
    if getattr(args, "profile", False):
        runner.complexity.summarize_profiles(output_dir)
    print(f"[Complete] {len(records)}/{len(items)} -> {output_dir}")


def path_from_here(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (HERE / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final Video QA method on LongVideoBench."
    )
    parser.add_argument("--annotation", required=True, help="LongVideoBench JSON or JSONL.")
    parser.add_argument("--video_dir", required=True, help="LongVideoBench video root.")
    parser.add_argument("--output_dir", default="outputs/longvideobench")
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record per-question complexity and planner statistics.",
    )
    parser.add_argument(
        "--question_ids_file",
        default=None,
        help="JSON/JSONL/TXT IDs to run, in the requested order.",
    )
    parser.add_argument(
        "--profile_cache_root",
        default=None,
        help="Independent cache root used only by profiling runs.",
    )
    parser.add_argument(
        "--profile_run_name",
        default=None,
        help="Unique directory name for one reproducible profiling run.",
    )
    args = parser.parse_args()
    if args.compact:
        args.save_diagnostics = False
    args.annotation = path_from_here(args.annotation)
    args.video_dir = path_from_here(args.video_dir)
    args.output_dir = path_from_here(args.output_dir)
    args.level_jsonl = path_from_here(args.level_jsonl) if args.level_jsonl else None
    args.question_ids_file = path_from_here(args.question_ids_file) if args.question_ids_file else None
    args.profile_cache_root = path_from_here(args.profile_cache_root) if args.profile_cache_root else None
    if args.profile_run_name:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.profile_run_name):
            parser.error("--profile_run_name must use only letters, digits, '.', '_' or '-'.")
    elif args.profile:
        parser.error("--profile requires an explicit --profile_run_name for unique, resumable outputs.")
    args.current_profile = None
    args.profile_seen_videos = set()
    args.dataset_format = "longvideobench"
    return args


def main() -> int:
    args = parse_args()
    if not args.annotation.is_file():
        raise FileNotFoundError(f"Annotation not found: {args.annotation}")
    if not args.video_dir.is_dir():
        raise NotADirectoryError(f"Video directory not found: {args.video_dir}")
    if args.level_jsonl is not None and not args.level_jsonl.is_file():
        raise FileNotFoundError(f"Level JSONL not found: {args.level_jsonl}")
    if not args.prepare_only and not args.api_key:
        raise RuntimeError(
            "Qwen API key is empty. Set DASHSCOPE_API_KEY/UNIFIED_QWEN_API_KEY "
            "or fill API_KEY in config.py."
        )

    items, missing, collisions = load_longvideobench_dataset(
        args.annotation, args.video_dir, args.level_jsonl
    )
    items = runner.select_question_ids(
        items, runner.load_question_ids(args.question_ids_file)
    )
    if collisions:
        preview = ", ".join(sorted(collisions)[:8])
        print(
            f"[Warning] {len(collisions)} duplicate video stems were found; "
            f"the first sorted path is used. Examples: {preview}"
        )
    if args.max_questions is not None:
        items = items[:max(0, args.max_questions)]
    if missing:
        preview = ", ".join(item["question_id"] for item in missing[:8])
        message = (
            f"{len(missing)} questions have no matching video; examples: {preview}"
        )
        if not args.allow_missing_videos:
            raise RuntimeError(
                message + ". Fix video_path/video_id or use --allow_missing_videos."
            )
        print("[Warning] " + message)
    if not items:
        raise RuntimeError("No runnable LongVideoBench questions remain after video matching.")

    option_counts = Counter(len(item["option_letters"]) for item in items)
    print(
        f"[LongVideoBench] runnable={len(items)}, missing={len(missing)}, "
        f"option_counts={dict(option_counts)}, frame_budgets={args.answer_frames}"
    )
    for frame_budget in args.answer_frames:
        run_budget(args, items, args.annotation, frame_budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# CUDA_VISIBLE_DEVICES=0 python scripts/run_our_method_longvideobench.py \
#   --annotation /path/to/LongVideoBench/lvb_val.json \
#   --video_dir /path/to/LongVideoBench/videos \
#   --output_dir ./outputs/longvideobench_prepare \
#   --routing_mode auto \
#   --allow_missing_videos \
#   --prepare_only


# CUDA_VISIBLE_DEVICES=0 python scripts/run_our_method_longvideobench.py \
#   --annotation /path/to/LongVideoBench/lvb_val.json \
#   --video_dir /path/to/LongVideoBench/videos \
#   --output_dir ./outputs/longvideobench \
#   --routing_mode auto \
#   --allow_missing_videos
