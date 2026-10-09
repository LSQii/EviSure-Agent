"""Optional, side-effect-free complexity profiling for the final QA runners.

The profiler is intentionally independent from the method's decision logic.  A
normal run does not import torch through this module and produces exactly the
legacy outputs.  A ``--profile`` run additionally writes per-question records,
planner records, a run manifest, and ready-to-use CSV/Markdown summaries.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _round(value: float) -> float:
    return round(float(value), 6)


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _safe_round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _read_jsonl_last_by_id(path: Path) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or "").strip()
            if qid:
                rows[qid] = row
    return list(rows.values())


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


class QuestionProfiler:
    """Collect exclusive stage timings and overlapping resource counters."""

    def __init__(
        self,
        question_id: str,
        video_id: str,
        dataset: str,
        method: str,
        cache_state: str,
        video_key: str | None = None,
    ) -> None:
        self.question_id = str(question_id)
        self.video_id = str(video_id)
        self.dataset = str(dataset)
        self.method = str(method)
        self.cache_state = str(cache_state)
        self.video_key = str(video_key or video_id)
        self.stage_seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.calls_by_stage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.resource_seconds_by_stage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.sampled_frames_by_stage: dict[str, int] = defaultdict(int)
        self.vlm_frame_exposures_by_stage: dict[str, int] = defaultdict(int)
        self.api_frame_exposures_by_stage: dict[str, int] = defaultdict(int)
        self.sampled_frame_keys: set[str] = set()
        self.cache: dict[str, Any] = {
            "state": cache_state,
            # The current pipeline does not cache decoded video frames.  Warm
            # means only that an earlier question from the same video has run;
            # component-level fields below state what was actually reused.
            "same_video_seen": cache_state == "warm",
            "video_cache_hit": False,
            "baseline_audio_hit": None,
            "extra_audio_hit": None,
            "extra_ocr_hit": None,
        }
        self._stage_stack: list[str] = []
        self._started = 0.0
        self._memory_before = 0.0
        self._peak_allocated = 0.0
        self._peak_reserved = 0.0

    @property
    def current_stage(self) -> str:
        return self._stage_stack[-1] if self._stage_stack else "unattributed"

    def start(self) -> None:
        _cuda_sync()
        self._started = time.perf_counter()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._memory_before = torch.cuda.memory_allocated() / 1024**3
        except Exception:
            pass

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        _cuda_sync()
        started = time.perf_counter()
        self._stage_stack.append(str(name))
        try:
            yield
        finally:
            _cuda_sync()
            self.stage_seconds[str(name)] += time.perf_counter() - started
            self._stage_stack.pop()

    def add_call(self, kind: str, count: int = 1, stage: str | None = None) -> None:
        stage_name = stage or self.current_stage
        self.calls[str(kind)] += int(count)
        self.calls_by_stage[stage_name][str(kind)] += int(count)

    def add_call_time(self, kind: str, seconds: float, stage: str | None = None) -> None:
        key = f"{kind}_wait_total"
        self.stage_seconds[key] += float(seconds)
        self.resource_seconds_by_stage[stage or self.current_stage][str(kind)] += float(seconds)

    def add_sampled_frames(self, count: int, timestamps: list[float] | None = None) -> None:
        stage = self.current_stage
        self.sampled_frames_by_stage[stage] += int(count)
        for value in timestamps or []:
            self.sampled_frame_keys.add(f"{self.video_id}:{float(value):.4f}")

    def add_vlm_exposure(self, count: int) -> None:
        self.vlm_frame_exposures_by_stage[self.current_stage] += int(count)

    def add_api_exposure(self, count: int) -> None:
        self.api_frame_exposures_by_stage[self.current_stage] += int(count)

    def set_cache(self, key: str, value: Any) -> None:
        self.cache[str(key)] = value

    def finish(
        self,
        result: dict[str, Any],
        initial_diag: dict[str, Any] | None,
        relation: dict[str, Any] | None,
        stability: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _cuda_sync()
        total = max(0.0, time.perf_counter() - self._started)
        try:
            import torch

            if torch.cuda.is_available():
                self._peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
                self._peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        except Exception:
            pass

        exclusive = sum(
            value
            for key, value in self.stage_seconds.items()
            if not key.endswith("_wait_total")
        )
        timing = {key: _round(value) for key, value in sorted(self.stage_seconds.items())}
        timing["total"] = _round(total)
        timing["unattributed_overhead"] = _round(max(0.0, total - exclusive))

        base_diag = (initial_diag or {}).get("base_evidence", {})
        base_answer = str(base_diag.get("final_answer") or "")
        base_prediction = _extract_letter(base_answer)
        ground_truth = str(result.get("ground_truth") or "")
        initial_planner = (initial_diag or {}).get("planner", {})
        relation_planner = (relation or {}).get("planner", {})

        sampled_total = sum(self.sampled_frames_by_stage.values())
        exposure_total = sum(self.vlm_frame_exposures_by_stage.values())
        api_exposure_total = sum(self.api_frame_exposures_by_stage.values())
        profile = {
            "question_id": self.question_id,
            "video_id": self.video_id,
            "video_key": self.video_key,
            "dataset": self.dataset,
            "method": self.method,
            "status": result.get("status", "unknown"),
            "execution_path": {
                "relation_target": bool(result.get("relation_target")),
                "reopen_ready": bool(result.get("reopen_ready")),
                "stability_executed": stability is not None,
                "verifier_route": result.get("verifier_route", "PASS_THROUGH"),
            },
            "cache": self.cache,
            "timing_seconds": timing,
            "calls": {
                **dict(sorted(self.calls.items())),
                "by_stage": {key: dict(sorted(value.items())) for key, value in sorted(self.calls_by_stage.items())},
            },
            "resource_seconds_by_stage": {
                key: {resource: _round(seconds) for resource, seconds in sorted(value.items())}
                for key, value in sorted(self.resource_seconds_by_stage.items())
            },
            "frames": {
                "global_answer": int((initial_diag or {}).get("global_frame_count", 0)),
                "local_answer": int((initial_diag or {}).get("local_frame_count", 0)),
                "initial_answer_input": int((initial_diag or {}).get("initial_answer_frame_count", 0)),
                "initial_storyboard": len(initial_planner.get("candidate_timestamps", []) or []),
                "relation_storyboard": len(relation_planner.get("storyboard_timestamps", []) or []),
                "audit_frames": len((relation or {}).get("audit_frames", {}).get("timestamps", []) or []),
                "sampled_frame_instances": sampled_total,
                "unique_sampled_timestamps": len(self.sampled_frame_keys),
                "vlm_frame_exposures": exposure_total,
                "api_frame_exposures": api_exposure_total,
                "total_visual_frame_exposures": exposure_total + api_exposure_total,
                "sampled_by_stage": dict(sorted(self.sampled_frames_by_stage.items())),
                "vlm_exposures_by_stage": dict(sorted(self.vlm_frame_exposures_by_stage.items())),
                "api_exposures_by_stage": dict(sorted(self.api_frame_exposures_by_stage.items())),
            },
            "memory_gb": {
                "before_question_allocated": _round(self._memory_before),
                "peak_allocated": _round(self._peak_allocated),
                "peak_reserved": _round(self._peak_reserved),
            },
            "checkpoints": {
                "base_videorag": {
                    "prediction": base_prediction,
                    "correct": base_prediction == ground_truth if ground_truth and base_prediction else None,
                    "final_frame_count": int(base_diag.get("final_frame_count", 0) or 0),
                },
                "after_planning": {
                    "prediction": result.get("initial_prediction", ""),
                    "correct": result.get("initial_prediction", "") == ground_truth if ground_truth else None,
                },
                "final": {
                    "prediction": result.get("final_prediction", ""),
                    "ground_truth": ground_truth,
                    "correct": result.get("correct"),
                },
            },
        }

        planner = {
            "question_id": self.question_id,
            "video_id": self.video_id,
            "dataset": self.dataset,
            "initial_planner": {
                "executed": bool(initial_planner),
                "status": initial_planner.get("api_status", "not_run"),
                "elapsed_seconds": timing.get("initial_planner", 0.0),
                "storyboard_sampling_seconds": _round(
                    self.resource_seconds_by_stage.get("initial_planner", {}).get("video_sampling", 0.0)
                ),
                "api_seconds": _round(
                    self.resource_seconds_by_stage.get("initial_planner", {}).get("api", 0.0)
                ),
                "api_calls": self.calls_by_stage.get("initial_planner", {}).get("api", 0),
                "storyboard_frames": len(initial_planner.get("candidate_timestamps", []) or []),
                "keyframe_valid": bool(initial_planner.get("keyframe_valid")),
                "proposed_anchor_frames": len(initial_planner.get("anchor_timestamps", []) or []),
                "actual_local_frames": int((initial_diag or {}).get("local_frame_count", 0)),
                "requested_tools": list(initial_planner.get("tool_requests", []) or []),
                "requested_asr": "ASR" in (initial_planner.get("tool_requests", []) or []),
                "requested_ocr": "OCR" in (initial_planner.get("tool_requests", []) or []),
                "requested_local_frames": "LOCAL_FRAMES" in (initial_planner.get("tool_requests", []) or []),
            },
            "relation_planner": {
                "executed": bool(relation_planner),
                "status": relation_planner.get("api_status", "not_run"),
                "parse_status": relation_planner.get("parse_status", "not_run"),
                "elapsed_seconds": timing.get("relation_planner", 0.0),
                "storyboard_sampling_seconds": _round(
                    self.resource_seconds_by_stage.get("relation_planner", {}).get("video_sampling", 0.0)
                ),
                "api_seconds": _round(
                    self.resource_seconds_by_stage.get("relation_planner", {}).get("api", 0.0)
                ),
                "api_calls": self.calls_by_stage.get("relation_planner", {}).get("api", 0),
                "route": (relation or {}).get("verifier_route", "PASS_THROUGH"),
                "storyboard_frames": len(relation_planner.get("storyboard_timestamps", []) or []),
                "audit_frames": len((relation or {}).get("audit_frames", {}).get("timestamps", []) or []),
            },
        }
        return profile, planner


def _extract_letter(value: Any) -> str:
    import re

    match = re.search(r"\b([A-E])\b", str(value or "").strip().upper())
    return match.group(1) if match else ""


def install_runtime_wrappers(baseline: Any, core: Any, args: Any) -> None:
    """Install wrappers once; they consult ``args.current_profile`` at call time."""

    if not getattr(baseline.llava_inference, "_complexity_profiled", False):
        original_llava = baseline.llava_inference

        def profiled_llava(question: Any, video: Any) -> Any:
            profile = getattr(args, "current_profile", None)
            if profile is not None:
                profile.add_call("vlm")
                count = 0
                try:
                    tensor = video[0] if isinstance(video, (list, tuple)) and video else video
                    count = int(tensor.shape[0]) if tensor is not None else 0
                except Exception:
                    count = 0
                profile.add_vlm_exposure(count)
            started = time.perf_counter()
            result = original_llava(question, video)
            if profile is not None:
                profile.add_call_time("vlm", time.perf_counter() - started)
            return result

        profiled_llava._complexity_profiled = True
        baseline.llava_inference = profiled_llava

    if not getattr(baseline.process_video, "_complexity_profiled", False):
        original_process_video = baseline.process_video

        def profiled_process_video(*call_args: Any, **call_kwargs: Any) -> Any:
            profile = getattr(args, "current_profile", None)
            started = time.perf_counter()
            result = original_process_video(*call_args, **call_kwargs)
            if profile is not None:
                profile.add_call_time("video_sampling", time.perf_counter() - started)
                frames = result[0] if isinstance(result, tuple) and result else []
                time_string = result[1] if isinstance(result, tuple) and len(result) > 1 else ""
                timestamps = []
                for raw in str(time_string or "").split(","):
                    raw = raw.strip().lower().removesuffix("s")
                    if raw:
                        try:
                            timestamps.append(float(raw))
                        except ValueError:
                            pass
                profile.add_sampled_frames(len(frames), timestamps)
            return result

        profiled_process_video._complexity_profiled = True
        baseline.process_video = profiled_process_video

    if not getattr(core.sample_unique_frames, "_complexity_profiled", False):
        original_sample_unique = core.sample_unique_frames

        def profiled_sample_unique(*call_args: Any, **call_kwargs: Any) -> Any:
            profile = getattr(args, "current_profile", None)
            started = time.perf_counter()
            result = original_sample_unique(*call_args, **call_kwargs)
            if profile is not None:
                profile.add_call_time("targeted_sampling", time.perf_counter() - started)
                frames = result[0] if isinstance(result, tuple) and result else []
                timestamps = result[1] if isinstance(result, tuple) and len(result) > 1 else []
                profile.add_sampled_frames(len(frames), list(timestamps))
            return result

        profiled_sample_unique._complexity_profiled = True
        core.sample_unique_frames = profiled_sample_unique

    for tool_name, counter_name in (("get_asr_docs", "asr"), ("get_ocr_docs", "ocr")):
        original = getattr(baseline, tool_name, None)
        if original is None or getattr(original, "_complexity_profiled", False):
            continue

        def make_tool_wrapper(function: Any, kind: str):
            def wrapped(*call_args: Any, **call_kwargs: Any) -> Any:
                profile = getattr(args, "current_profile", None)
                if profile is not None:
                    profile.add_call(kind)
                started = time.perf_counter()
                result = function(*call_args, **call_kwargs)
                if profile is not None:
                    profile.add_call_time(kind, time.perf_counter() - started)
                return result

            wrapped._complexity_profiled = True
            return wrapped

        setattr(baseline, tool_name, make_tool_wrapper(original, counter_name))

    if not getattr(core.api_chat, "_complexity_profiled", False):
        original_api_chat = core.api_chat

        def profiled_api_chat(*call_args: Any, **call_kwargs: Any) -> Any:
            profile = getattr(args, "current_profile", None)
            if profile is not None:
                profile.add_call("api")
                frames = call_kwargs.get("frames")
                if frames is None and len(call_args) > 4:
                    frames = call_args[4]
                try:
                    profile.add_api_exposure(len(frames) if frames is not None else 0)
                except Exception:
                    profile.add_api_exposure(0)
            started = time.perf_counter()
            try:
                return original_api_chat(*call_args, **call_kwargs)
            finally:
                if profile is not None:
                    profile.add_call_time("api", time.perf_counter() - started)

        profiled_api_chat._complexity_profiled = True
        core.api_chat = profiled_api_chat


def write_run_manifest(
    output_dir: Path,
    args: Any,
    dataset: str,
    frame_budget: int,
    model_load_seconds: float,
) -> None:
    payload: dict[str, Any] = {
        "dataset": dataset,
        "method": "our_method",
        "answer_frames": frame_budget,
        "planner_frames": getattr(args, "planner_frames", None),
        "relation_storyboard_frames": getattr(args, "relation_storyboard_frames", None),
        "api_base": getattr(args, "api_base", ""),
        "api_model": getattr(args, "api_model", ""),
        "routing_mode": getattr(args, "routing_mode", ""),
        "question_ids_file": str(getattr(args, "question_ids_file", "") or ""),
        "profile_cache_root": str(getattr(args, "profile_cache_root", "") or ""),
        "effective_profile_cache_root": str(getattr(args, "current_profile_cache_root", "") or ""),
        "profile_run_name": str(getattr(args, "profile_run_name", "") or ""),
        "model_load_seconds": _round(model_load_seconds),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    try:
        import torch

        payload.update({
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_resident_allocated_gb": _round(torch.cuda.memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
            "model_resident_reserved_gb": _round(torch.cuda.memory_reserved() / 1024**3) if torch.cuda.is_available() else 0.0,
        })
    except Exception as exc:
        payload["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _flatten_stage_counts(row: dict[str, Any], stage: str, kind: str) -> int:
    return int(row.get("calls", {}).get("by_stage", {}).get(stage, {}).get(kind, 0) or 0)


def summarize_profiles(output_dir: Path) -> None:
    rows = _read_jsonl_last_by_id(output_dir / "profile_records.jsonl")
    if not rows:
        return
    summary_dir = output_dir / "complexity_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    usable = [row for row in rows if row.get("status") != "error"]

    def values(path: tuple[str, ...], subset: list[dict[str, Any]] = usable) -> list[float]:
        result = []
        for row in subset:
            value: Any = row
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                result.append(float(value))
        return result

    total_times = values(("timing_seconds", "total"))
    cold = [row for row in usable if row.get("cache", {}).get("state") == "cold"]
    warm = [row for row in usable if row.get("cache", {}).get("state") == "warm"]
    final_correct = [bool(row.get("checkpoints", {}).get("final", {}).get("correct")) for row in usable]
    base_scored = [row for row in usable if row.get("checkpoints", {}).get("base_videorag", {}).get("correct") is not None]
    base_correct = [bool(row["checkpoints"]["base_videorag"]["correct"]) for row in base_scored]

    main_rows = [
        {
            "method": "Video-RAG (embedded checkpoint)",
            "n": len(base_scored),
            "accuracy": _safe_round(sum(base_correct) / len(base_correct) if base_correct else None),
            "mean_seconds": _safe_round(_mean(values(("timing_seconds", "base_videorag"), base_scored))),
            "median_seconds": _safe_round(_percentile(values(("timing_seconds", "base_videorag"), base_scored), 0.5)),
            "p95_seconds": _safe_round(_percentile(values(("timing_seconds", "base_videorag"), base_scored), 0.95)),
            "api_calls_per_question": _safe_round(_mean([_flatten_stage_counts(row, "base_videorag", "api") for row in base_scored])),
            "vlm_calls_per_question": _safe_round(_mean([_flatten_stage_counts(row, "base_videorag", "vlm") for row in base_scored])),
            "sampled_frames_per_question": _safe_round(_mean([float(row.get("frames", {}).get("sampled_by_stage", {}).get("base_videorag", 0)) for row in base_scored])),
            "vlm_frame_exposures_per_question": _safe_round(_mean([float(row.get("frames", {}).get("vlm_exposures_by_stage", {}).get("base_videorag", 0)) for row in base_scored])),
            "api_frame_exposures_per_question": _safe_round(_mean([float(row.get("frames", {}).get("api_exposures_by_stage", {}).get("base_videorag", 0)) for row in base_scored])),
            "total_visual_frame_exposures_per_question": _safe_round(_mean([float(row.get("frames", {}).get("vlm_exposures_by_stage", {}).get("base_videorag", 0)) + float(row.get("frames", {}).get("api_exposures_by_stage", {}).get("base_videorag", 0)) for row in base_scored])),
            "peak_allocated_gb": None,
            "cold_mean_seconds": None,
            "warm_mean_seconds": None,
        },
        {
            "method": "Our Method",
            "n": len(usable),
            "accuracy": _safe_round(sum(final_correct) / len(final_correct) if final_correct else None),
            "mean_seconds": _safe_round(_mean(total_times)),
            "median_seconds": _safe_round(_percentile(total_times, 0.5)),
            "p95_seconds": _safe_round(_percentile(total_times, 0.95)),
            "api_calls_per_question": _safe_round(_mean(values(("calls", "api")))),
            "vlm_calls_per_question": _safe_round(_mean(values(("calls", "vlm")))),
            "sampled_frames_per_question": _safe_round(_mean(values(("frames", "sampled_frame_instances")))),
            "vlm_frame_exposures_per_question": _safe_round(_mean(values(("frames", "vlm_frame_exposures")))),
            "api_frame_exposures_per_question": _safe_round(_mean(values(("frames", "api_frame_exposures")))),
            "total_visual_frame_exposures_per_question": _safe_round(_mean(values(("frames", "total_visual_frame_exposures")))),
            "peak_allocated_gb": _safe_round(max(values(("memory_gb", "peak_allocated")), default=0.0)),
            "cold_mean_seconds": _safe_round(_mean(values(("timing_seconds", "total"), cold))),
            "warm_mean_seconds": _safe_round(_mean(values(("timing_seconds", "total"), warm))),
        },
    ]
    main_fields = list(main_rows[0])
    with (summary_dir / "table1_main_complexity.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=main_fields)
        writer.writeheader()
        writer.writerows(main_rows)

    timing_keys = sorted({key for row in usable for key in row.get("timing_seconds", {}) if key not in {"total", "unattributed_overhead"} and not key.endswith("_wait_total")})
    stage_rows = []
    for stage in timing_keys:
        executed = [row for row in usable if float(row.get("timing_seconds", {}).get(stage, 0) or 0) > 0]
        stage_values = [float(row["timing_seconds"][stage]) for row in executed]
        stage_rows.append({
            "stage": stage,
            "executed_questions": len(executed),
            "execution_rate": _safe_round(len(executed) / len(usable) if usable else None),
            "mean_when_executed_seconds": _safe_round(_mean(stage_values)),
            "mean_per_all_questions_seconds": _safe_round(sum(stage_values) / len(usable) if usable else None),
            "p95_when_executed_seconds": _safe_round(_percentile(stage_values, 0.95)),
            "api_calls_per_all_questions": _safe_round(sum(_flatten_stage_counts(row, stage, "api") for row in usable) / len(usable) if usable else None),
            "vlm_calls_per_all_questions": _safe_round(sum(_flatten_stage_counts(row, stage, "vlm") for row in usable) / len(usable) if usable else None),
        })
    stage_fields = list(stage_rows[0]) if stage_rows else []
    if stage_rows:
        with (summary_dir / "stage_timing_table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=stage_fields)
            writer.writeheader()
            writer.writerows(stage_rows)

    planner_records = _read_jsonl_last_by_id(output_dir / "planner_records.jsonl")
    planner_rows = []
    for planner_name in ("initial_planner", "relation_planner"):
        executed = [row.get(planner_name, {}) for row in planner_records if row.get(planner_name, {}).get("executed")]
        elapsed = [float(row.get("elapsed_seconds", 0) or 0) for row in executed]
        success = [row for row in executed if row.get("status") == "ok"]
        planner_rows.append({
            "planner": planner_name,
            "executed_questions": len(executed),
            "execution_rate": _safe_round(len(executed) / len(planner_records) if planner_records else None),
            "success_rate": _safe_round(len(success) / len(executed) if executed else None),
            "mean_seconds_when_executed": _safe_round(_mean(elapsed)),
            "mean_storyboard_sampling_seconds": _safe_round(
                _mean([float(row.get("storyboard_sampling_seconds", 0) or 0) for row in executed])
            ),
            "mean_api_seconds": _safe_round(
                _mean([float(row.get("api_seconds", 0) or 0) for row in executed])
            ),
            "p95_seconds_when_executed": _safe_round(_percentile(elapsed, 0.95)),
            "api_calls_per_question": _safe_round(sum(int(row.get("api_calls", 0) or 0) for row in executed) / len(planner_records) if planner_records else None),
            "mean_storyboard_frames": _safe_round(_mean([float(row.get("storyboard_frames", 0) or 0) for row in executed])),
            "mean_output_frames": _safe_round(_mean([float(row.get("actual_local_frames", row.get("audit_frames", 0)) or 0) for row in executed])),
        })
    with (summary_dir / "table2_planner_stages.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(planner_rows[0]))
        writer.writeheader()
        writer.writerows(planner_rows)

    cold_times = values(("timing_seconds", "total"), cold)
    warm_times = values(("timing_seconds", "total"), warm)
    cold_mean, warm_mean = _mean(cold_times), _mean(warm_times)
    cache_row = {
        "cold_n": len(cold),
        "warm_n": len(warm),
        "cold_mean_seconds": _safe_round(cold_mean),
        "warm_mean_seconds": _safe_round(warm_mean),
        "warm_speedup": _safe_round(cold_mean / warm_mean if cold_mean and warm_mean else None),
        "baseline_audio_hit_rate": _safe_round(_mean([float(bool(row.get("cache", {}).get("baseline_audio_hit"))) for row in usable if row.get("cache", {}).get("baseline_audio_hit") is not None])),
        "extra_audio_hit_rate": _safe_round(_mean([float(bool(row.get("cache", {}).get("extra_audio_hit"))) for row in usable if row.get("cache", {}).get("extra_audio_hit") is not None])),
        "extra_ocr_hit_rate": _safe_round(_mean([float(bool(row.get("cache", {}).get("extra_ocr_hit"))) for row in usable if row.get("cache", {}).get("extra_ocr_hit") is not None])),
    }
    with (summary_dir / "cache_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cache_row))
        writer.writeheader()
        writer.writerow(cache_row)

    report = ["# Complexity profiling report", "", f"Usable questions: {len(usable)}/{len(rows)}", "", "## Table 1", ""]
    report.extend(_markdown_table(main_fields, main_rows))
    report.extend(["", "## Table 2: planners", ""])
    report.extend(_markdown_table(list(planner_rows[0]), planner_rows))
    report.extend(["", "## Cache", ""])
    report.extend(_markdown_table(list(cache_row), [cache_row]))
    report.extend([
        "",
        "> The embedded Video-RAG checkpoint is suitable for paired accuracy and stage-latency analysis. "
        "Its peak VRAM is intentionally left blank because the full method's modules are resident in the same process.",
    ])
    (summary_dir / "complexity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _markdown_table(fields: list[str], rows: list[dict[str, Any]]) -> list[str]:
    result = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        result.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return result
