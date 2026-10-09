"""Full evidence-gap routing and counter-evidence reflection experiment.

Default run (all questions):
    python evidence_gap_reflective_agent.py

Fill QWEN_API_KEY below before running.  The experiment freezes each original
Video-RAG evidence prompt, asks Qwen to inspect a denser storyboard and route
only missing evidence, lets the local LLaVA answer independently, and finally
uses one bounded counter-evidence critique.  The Qwen candidate answer is
recorded as a control but is never passed to LLaVA.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Fill locally only when environment variables are unavailable.


QWEN_API_KEY = ""
QWEN_API_BASE = "https://YOUR_COMPATIBLE_MODE_ENDPOINT/v1"

QWEN_API_MODEL = "qwen3-vl-plus"

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE.parent / "data" / "video"
DEFAULT_ANNOTATION = HERE.parent / "data" / "Video-MME.json"
DEFAULT_LEVELS = HERE / "question_level_results" / "question_levels_20260629_223422.jsonl"
DEFAULT_BASELINE_SEED = HERE / "result" / "smoke_baseline" / "eval_details.jsonl"
DEFAULT_OUTPUT_DIR = HERE / "result" / "evidence_gap_reflective"

PROTOCOL_VERSION = "evidence-gap-reflective"
PLANNER_PROTOCOL_VERSION = "planner-grounded-frames"
TOOL_PROTOCOL_VERSION = "tools-raw-asr"
CRITIC_PROTOCOL_VERSION = "critic-grounded-visual-counterevidence"
EXPECTED_FULL_QUESTIONS = 1368
GLOBAL_FRAMES = 16
PLANNER_FRAMES = 32
DENSE_OCR_FRAMES = 32
MAX_LOCAL_FRAMES = 8
MAX_LOCAL_WINDOW_SECONDS = 12.0
IMAGE_MAX_SIDE = 512
RAG_THRESHOLD = 0.3
API_MAX_RETRIES = 5
API_RETRY_WAIT = 60.0
ANSWER_MARKER = "Select the best answer to the following multiple-choice question"
METHODS = (
    "baseline_reference",
    "baseline_paired",
    "qwen_planner_candidate",
    "agent_pre_reflection",
    "agent_final",
)


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
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


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def stable_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_manifest_hash(data_dir: Path) -> str:
    names = []
    for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        names.extend(path.name for path in data_dir.glob(pattern))
    return stable_hash(sorted(names))


def cache_files_exist(output_dir: Path) -> bool:
    names = (
        "generated_baseline_cache.jsonl", "planner_cache.jsonl", "tool_cache.jsonl",
        "critic_cache.jsonl", "results.jsonl",
    )
    return any((output_dir / name).exists() and (output_dir / name).stat().st_size > 0 for name in names)


def build_video_map(data_dir: Path) -> dict[str, str]:
    result = {}
    for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        for path in data_dir.glob(pattern):
            result[path.stem] = str(path)
    return result


def youtube_id(url: str) -> str | None:
    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return None


def load_levels(path: Path) -> dict[str, str]:
    result = {}
    for qid, row in load_jsonl(path).items():
        level = str(row.get("question_level") or row.get("level") or "UNKNOWN").upper()
        result[qid] = level if level in {"A", "B", "C"} else "UNKNOWN"
    return result


def build_question_list(annotation: Path, data_dir: Path, level_path: Path) -> list[dict]:
    dataset = json.loads(annotation.read_text(encoding="utf-8"))
    videos = build_video_map(data_dir)
    levels = load_levels(level_path)
    result = []
    for video in dataset:
        path = videos.get(youtube_id(str(video.get("url") or "")) or "")
        if not path:
            continue
        for question in video.get("questions", []):
            qid = str(question.get("question_id") or "").strip()
            if not qid:
                continue
            result.append({
                "video_id": str(video.get("video_id") or ""),
                "video_path": path,
                "question_id": qid,
                "question_level": levels.get(qid, "UNKNOWN"),
                "task_type": str(question.get("task_type") or "Unknown"),
                "video_category": str(video.get("video_category") or "Unknown"),
                "video_subcategory": str(video.get("video_subcategory") or "Unknown"),
                "question": str(question.get("question") or "") + " " + " ".join(question.get("choices", [])),
                "ground_truth": str(question.get("answer") or "").strip().upper(),
            })
    return result


def parse_json_loose(text: str) -> dict:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        raw = match.group(0)
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def extract_letter(value) -> str:
    text = str(value or "").strip().upper()
    match = re.search(r"(?:OPTION|ANSWER|CHOICE)[\s_:\-]*([A-D])", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1) if match else ""


def normalize_list(value) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def normalize_tools(value) -> list[str]:
    aliases = {
        "ASR": "ASR",
        "SPEECH": "ASR",
        "OCR": "OCR",
        "TEXT": "OCR",
        "LOCAL_FRAMES": "LOCAL_FRAMES",
        "KEYFRAMES": "LOCAL_FRAMES",
        "LOCAL_VISUAL": "LOCAL_FRAMES",
    }
    result = []
    for item in normalize_list(value):
        tool = aliases.get(item.upper())
        if tool and tool not in result:
            result.append(tool)
    return result[:2]


def normalize_indices(value, count: int) -> list[int]:
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= index < count and index not in result:
            result.append(index)
    return result[:3]


def normalize_budget(value, default: int = 0) -> int:
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = default
    return max((x for x in (0, 2, 4, 6, 8) if x <= requested), default=0)


def clean_visual_evidence(value) -> list[str]:
    result = []
    for text in normalize_list(value):
        # Candidate labels must not leak from Qwen into the local answer model.
        text = re.sub(r"\b(?:option|answer|choice)[\s_:\-]*[A-D]\b", "a candidate option", text, flags=re.I)
        text = text.strip()
        if text and text not in result:
            result.append(text)
    return result[:4]


def clean_grounded_audit_evidence(value) -> list[str]:
    result = []
    provenance = re.compile(r"(?:BASELINE_(?:ASR|OCR|DET)|\bASR\b|\bOCR\b|\bDET\b|\b[EL]\d+\b|\d+(?:\.\d+)?s)", re.I)
    for text in normalize_list(value):
        if not provenance.search(text):
            continue
        text = re.sub(r"\b(?:option|answer|choice)[\s_:\-]*[A-D]\b", "a candidate option", text, flags=re.I)
        text = text.strip()
        if text and text not in result:
            result.append(text)
    return result[:8]


def frame_to_data_url(frame, max_side: int = IMAGE_MAX_SIDE) -> str:
    image = Image.fromarray(frame.astype("uint8")).convert("RGB")
    width, height = image.size
    scale = min(1.0, float(max_side) / max(width, height))
    if scale < 1.0:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def label_frames(frames, timestamps, prefix: str = "C") -> list[np.ndarray]:
    result = []
    for index, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        image = Image.fromarray(frame.astype("uint8")).convert("RGB")
        bar = max(24, min(40, image.height // 12))
        canvas = Image.new("RGB", (image.width, image.height + bar), "black")
        canvas.paste(image, (0, bar))
        ImageDraw.Draw(canvas).text((8, 6), f"{prefix}{index:02d} | {timestamp:.2f}s", fill="white")
        result.append(np.asarray(canvas))
    return result


class ApiContentBlocked(RuntimeError):
    """The remote API rejected supplied content during data inspection."""


def api_chat(api_key: str, api_base: str, api_model: str, prompt: str, frames=None, max_tokens: int = 512) -> str:
    content = [{"type": "text", "text": prompt}]
    for frame in frames or []:
        content.append({"type": "image_url", "image_url": {"url": frame_to_data_url(frame)}})
    payload = {
        "model": api_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    last_error = None
    for attempt in range(API_MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body}"
            body_lower = body.lower()
            if exc.code == 400 and (
                "data_inspection_failed" in body_lower
                or "inappropriate content" in body_lower
            ):
                raise ApiContentBlocked(last_error) from exc
            if exc.code in {429, 500, 502, 503, 504} and attempt + 1 < API_MAX_RETRIES:
                time.sleep(API_RETRY_WAIT)
                continue
            raise RuntimeError(last_error) from exc
        except Exception as exc:
            last_error = str(exc)
            if attempt + 1 < API_MAX_RETRIES:
                time.sleep(API_RETRY_WAIT)
                continue
            raise RuntimeError(last_error) from exc
    raise RuntimeError(last_error or "api_failed")


def planner_prompt(question: str, level: str, timestamps: list[float], global_frames: int = GLOBAL_FRAMES) -> str:
    labels = ", ".join(f"C{i:02d}={ts:.2f}s" for i, ts in enumerate(timestamps))
    return f"""You are the evidence-gap planner for a long-video multiple-choice QA agent.
Inspect the chronological labelled storyboard and the question. First form a private
candidate answer, then decide what evidence is still missing. Do not assume the
candidate is correct. The candidate is recorded only as a control and will not be
shown to the final answer model.

Question level: {level}
Question and options: {question}
Storyboard timestamps: {labels}

Available incremental tools:
- ASR: spoken narration/dialogue;
- OCR: visible slides, captions, scoreboards, signs;
- LOCAL_FRAMES: 2/4/6/8 additional frames around up to three visible anchors.
The original Video-RAG {global_frames} global frames and evidence are always retained.

Rules:
- Request at most two tools and only for evidence missing from the storyboard.
- Use LOCAL_FRAMES only for a local action, before/after relation, state transition,
  or small visual detail with a visible anchor.
- Global narrative, causality explained by speech, and counting across the whole
  video are not reliable local-frame tasks.
- observed_visual_evidence must contain visible facts with timestamps and must not
  mention answer/option letters.
- visual_evidence_frame_indices must identify up to four storyboard frames that
  visibly ground observed_visual_evidence, even when no nearby sampling is needed.
- Even if you have a likely answer, request a tool when decisive evidence is absent.
- keyframe_mode=protect_only means selected frames themselves are sufficient;
  keyframe_mode=local_window means anchors plus nearby frames are required.
- local_frame_budget is the total number of additional frames, including anchors.

Return JSON only:
{{
  "candidate_answer": "A/B/C/D",
  "candidate_confidence": "high/medium/low",
  "observed_visual_evidence": ["timestamped visible fact without option letters"],
  "visual_evidence_frame_indices": [0],
  "evidence_sufficient": false,
  "missing_evidence": "speech/text/local_transition/local_detail/global_context/none",
  "tool_requests": ["ASR", "OCR", "LOCAL_FRAMES"],
  "asr_query": "specific spoken fact to retrieve or empty",
  "ocr_keywords": ["specific visible words"],
  "need_local_frames": true,
  "keyframe_mode": "none/protect_only/local_window",
  "anchor_frame_indices": [0],
  "local_window": "before/after/around/none",
  "local_frame_budget": 8,
  "reason": "why these evidence sources are needed"
}}"""


def planner_request_signature(
    item: dict, api_base: str, api_model: str, planner_frames: int, global_frames: int = GLOBAL_FRAMES
) -> str:
    return stable_hash({
        "protocol": PLANNER_PROTOCOL_VERSION,
        "question": item["question"],
        "question_level": item["question_level"],
        "video_stem": Path(item["video_path"]).stem,
        "api_base": api_base,
        "api_model": api_model,
        "planner_frames": planner_frames,
        "global_frames": global_frames,
    })


def run_planner(
    baseline, item: dict, api_key: str, api_base: str, api_model: str, planner_frames: int,
    global_frames: int = GLOBAL_FRAMES,
) -> dict:
    frames, time_string, duration = baseline.process_video(item["video_path"], planner_frames, 1, force_sample=True)
    timestamps = [float(x.replace("s", "")) for x in time_string.split(",")]
    try:
        raw = api_chat(
            api_key,
            api_base,
            api_model,
            planner_prompt(item["question"], item["question_level"], timestamps, global_frames),
            label_frames(frames, timestamps),
            max_tokens=640,
        )
    except ApiContentBlocked as exc:
        print(f"[Qwen content blocked] planner {item['question_id']}; falling back to original Video-RAG.")
        return {
            "question_id": item["question_id"],
            "request_signature": planner_request_signature(item, api_base, api_model, planner_frames, global_frames),
            "api_status": "content_blocked",
            "api_error": str(exc),
            "raw_response": "",
            "parsed": {},
            "candidate_answer": "",
            "candidate_confidence": "low",
            "observed_visual_evidence": [],
            "visual_evidence_frame_indices": [],
            "evidence_sufficient": False,
            "missing_evidence": "api_content_blocked",
            "tool_requests": [],
            "asr_query": "",
            "ocr_keywords": [],
            "candidate_timestamps": timestamps,
            "duration": float(duration),
            "anchor_frame_indices": [],
            "anchor_timestamps": [],
            "keyframe_mode": "none",
            "local_window": "none",
            "local_frame_budget": 0,
            "keyframe_valid": False,
            "reason": "Qwen rejected one or more storyboard frames during content inspection.",
        }
    parsed = parse_json_loose(raw)
    tools = normalize_tools(parsed.get("tool_requests"))
    missing_evidence = str(parsed.get("missing_evidence") or "none").lower()
    evidence_sufficient = parse_bool(parsed.get("evidence_sufficient"))
    # A malformed/under-specified API response must not silently drop the
    # modality it explicitly says is missing.
    inferred_tool = {
        "speech": "ASR",
        "text": "OCR",
        "local_transition": "LOCAL_FRAMES",
        "local_detail": "LOCAL_FRAMES",
    }.get(missing_evidence)
    if not evidence_sufficient and inferred_tool and inferred_tool not in tools and len(tools) < 2:
        tools.append(inferred_tool)
    visual_indices = normalize_indices(parsed.get("visual_evidence_frame_indices"), len(timestamps))
    need_local = parse_bool(parsed.get("need_local_frames")) or "LOCAL_FRAMES" in tools
    if need_local and "LOCAL_FRAMES" not in tools and len(tools) < 2:
        tools.append("LOCAL_FRAMES")
    indices = normalize_indices(parsed.get("anchor_frame_indices"), len(timestamps))
    if not indices and "LOCAL_FRAMES" in tools:
        indices = list(visual_indices[:3])
    keyframe_mode = str(parsed.get("keyframe_mode") or "none").strip().lower()
    if keyframe_mode not in {"none", "protect_only", "local_window"}:
        keyframe_mode = "none"
    if keyframe_mode != "none" and indices and "LOCAL_FRAMES" not in tools and len(tools) < 2:
        tools.append("LOCAL_FRAMES")
    window = str(parsed.get("local_window") or "none").lower()
    if window not in {"before", "after", "around"}:
        window = "none"
    if "LOCAL_FRAMES" in tools and keyframe_mode == "none":
        keyframe_mode = "local_window" if window != "none" else "protect_only"
    if keyframe_mode == "local_window" and window == "none":
        window = "around"
    budget_default = 8 if keyframe_mode == "local_window" else (2 if keyframe_mode == "protect_only" else 0)
    budget = normalize_budget(parsed.get("local_frame_budget"), default=budget_default)
    if keyframe_mode != "none" and budget == 0:
        budget = budget_default
    keyframe_valid = "LOCAL_FRAMES" in tools and bool(indices) and keyframe_mode != "none" and budget > 0
    return {
        "question_id": item["question_id"],
        "request_signature": planner_request_signature(item, api_base, api_model, planner_frames, global_frames),
        "api_status": "ok",
        "api_error": "",
        "raw_response": raw,
        "parsed": parsed,
        "candidate_answer": extract_letter(parsed.get("candidate_answer")),
        "candidate_confidence": str(parsed.get("candidate_confidence") or "low").lower(),
        "observed_visual_evidence": clean_visual_evidence(parsed.get("observed_visual_evidence")),
        "visual_evidence_frame_indices": visual_indices,
        "evidence_sufficient": evidence_sufficient,
        "missing_evidence": missing_evidence,
        "tool_requests": tools,
        "asr_query": str(parsed.get("asr_query") or "").strip(),
        "ocr_keywords": normalize_list(parsed.get("ocr_keywords"))[:8],
        "candidate_timestamps": timestamps,
        "duration": float(duration),
        "anchor_frame_indices": indices,
        "anchor_timestamps": [timestamps[index] for index in indices],
        "keyframe_mode": keyframe_mode,
        "local_window": window,
        "local_frame_budget": budget,
        "keyframe_valid": keyframe_valid,
        "reason": str(parsed.get("reason") or "").strip(),
    }


def allocate_budget(anchor_count: int, budget: int) -> list[int]:
    if anchor_count <= 0 or budget <= 0:
        return []
    base, remainder = divmod(budget, anchor_count)
    return [base + (1 if i < remainder else 0) for i in range(anchor_count)]


def proposed_keyframe_timestamps(planner: dict, force: bool = False) -> list[float]:
    anchors = planner.get("anchor_timestamps", [])[:3]
    if not anchors:
        return []
    duration = float(planner.get("duration") or 0.0)
    window = planner.get("local_window") if planner.get("local_window") in {"before", "after", "around"} else "around"
    budget = normalize_budget(planner.get("local_frame_budget"), default=8)
    if force:
        budget = 8
    if budget <= 0:
        return []
    # The selected anchors are the actual keyframes.  They count toward the
    # total dynamic budget and must reach the final model before nearby frames.
    result = []
    for anchor in anchors:
        if anchor not in result and len(result) < budget:
            result.append(float(anchor))
    if (planner.get("keyframe_mode") == "protect_only" and not force) or len(result) >= budget:
        return sorted(result)

    remaining = budget - len(result)
    radius = min(MAX_LOCAL_WINDOW_SECONDS, max(1.0, duration / max(1, len(planner.get("candidate_timestamps", [])))))
    for anchor, count in zip(anchors, allocate_budget(len(anchors), remaining)):
        if window == "before":
            values = np.linspace(max(0.0, anchor - radius), anchor, count + 1)[:-1]
        elif window == "after":
            values = np.linspace(anchor, min(duration, anchor + radius), count + 1)[1:]
        else:
            before_count = (count + 1) // 2
            after_count = count - before_count
            before_values = (
                np.linspace(max(0.0, anchor - radius), anchor, before_count + 1)[:-1]
                if before_count else []
            )
            after_values = (
                np.linspace(anchor, min(duration, anchor + radius), after_count + 1)[1:]
                if after_count else []
            )
            values = list(before_values) + list(after_values)
        for value in values:
            timestamp = float(value)
            if all(abs(timestamp - existing) > 1e-6 for existing in result):
                result.append(timestamp)
    return sorted(result)[:budget]


def sample_unique_frames(video_path: str, timestamps: list[float], global_timestamps: list[float]):
    if not timestamps:
        return np.zeros((0, 336, 336, 3), dtype=np.uint8), []
    from decord import VideoReader, cpu

    reader = VideoReader(video_path, ctx=cpu(), num_threads=1)
    try:
        fps = float(reader.get_avg_fps())
        total = len(reader)
        blocked = {min(max(int(ts * fps), 0), total - 1) for ts in global_timestamps}
        indices, actual = [], []
        for timestamp in timestamps:
            index = min(max(int(float(timestamp) * fps), 0), total - 1)
            if index in blocked or index in indices:
                continue
            indices.append(index)
            actual.append(index / fps)
        if not indices:
            return np.zeros((0, 336, 336, 3), dtype=np.uint8), []
        return reader.get_batch(indices).asnumpy(), actual
    finally:
        del reader


def stamp_frames(frames, timestamps: list[float], prefix: str, start_index: int = 0) -> np.ndarray:
    stamped = []
    for index, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        image = Image.fromarray(frame.astype("uint8")).convert("RGB")
        draw = ImageDraw.Draw(image)
        label = f"{prefix}{start_index + index + 1} | {timestamp:.2f}s"
        draw.rectangle((0, 0, min(image.width, 220), 26), fill="black")
        draw.text((6, 6), label, fill="white")
        stamped.append(np.asarray(image))
    return np.stack(stamped, axis=0) if stamped else np.zeros((0, 336, 336, 3), dtype=np.uint8)


def planner_evidence_frames(baseline, item: dict, planner: dict, planner_frame_count: int):
    indices = []
    for index in planner.get("visual_evidence_frame_indices", []) + planner.get("anchor_frame_indices", []):
        if index not in indices and 0 <= int(index) < planner_frame_count:
            indices.append(int(index))
    indices = sorted(indices[:4])
    if not indices:
        return np.zeros((0, 336, 336, 3), dtype=np.uint8), []
    frames, time_string, _ = baseline.process_video(item["video_path"], planner_frame_count, 1, force_sample=True)
    timestamps = [float(x.replace("s", "")) for x in time_string.split(",")]
    valid = [index for index in indices if index < len(frames)]
    selected_ts = [timestamps[index] for index in valid]
    selected_frames = frames[valid]
    return stamp_frames(selected_frames, selected_ts, "E"), selected_ts


def retrieve_asr(baseline, item: dict, query: str, cache_root: Path | None = None) -> dict:
    cache_dir = (cache_root / "video_asr_cache") if cache_root is not None else (HERE / "restore" / "audio")
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(item["video_path"]).resolve()
    cache_key = stable_hash(str(video_path))[:16]
    stem = video_path.stem
    text_path = cache_dir / f"{stem}_{cache_key}.txt"
    if text_path.exists():
        raw_docs = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
    else:
        audio_path = cache_dir / f"{stem}_{cache_key}.wav"
        raw_docs = baseline.get_asr_docs(item["video_path"], str(audio_path))
        text_path.write_text("\n".join(raw_docs), encoding="utf-8")
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass
    cleaned_docs, time_ranges = [], []
    for doc_index, document in enumerate(raw_docs):
        text = str(document).strip()
        if not text:
            continue
        # Some caches already contain timestamp prefixes.  Strip them before
        # semantic retrieval, but preserve the real range as provenance.
        match = re.match(r"^\[(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\]\s*(.*)$", text, flags=re.I)
        if match:
            start, end, text = float(match.group(1)), float(match.group(2)), match.group(3).strip()
        else:
            start, end = doc_index * 30.0, (doc_index + 1) * 30.0
        if text:
            cleaned_docs.append(text)
            time_ranges.append((start, end))
    queries = [item["question"]] + ([query] if query else [])
    docs, indices = baseline.retrieve_documents_with_dynamic(cleaned_docs, queries, threshold=RAG_THRESHOLD) if cleaned_docs else ([], [])
    formatted = []
    for document, index in zip(docs, indices):
        start, end = time_ranges[index]
        formatted.append(f"[ASR chunk {index}, {start:.2f}-{end:.2f}s] {document}")
    return {"documents": formatted, "raw_document_count": len(cleaned_docs), "retrieved_count": len(formatted)}


def load_or_build_dense_ocr(baseline, item: dict, output_dir: Path) -> dict:
    cache_dir = output_dir / "video_ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(item["video_path"]).resolve()
    cache_key = stable_hash(str(video_path))[:16]
    path = cache_dir / f"{video_path.stem}_{cache_key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    frames, time_string, _ = baseline.process_video(item["video_path"], DENSE_OCR_FRAMES, 1, force_sample=True)
    timestamps = [float(x.replace("s", "")) for x in time_string.split(",")]
    docs, frame_indices = baseline.get_ocr_docs(frames)
    payload = {"documents": docs, "frame_indices": frame_indices, "timestamps": timestamps}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def retrieve_ocr(baseline, item: dict, keywords: list[str], output_dir: Path) -> dict:
    cache = load_or_build_dense_ocr(baseline, item, output_dir)
    docs = cache.get("documents", [])
    queries = [item["question"]] + keywords
    retrieved, indices = baseline.retrieve_documents_with_dynamic(docs, queries, threshold=RAG_THRESHOLD) if docs else ([], [])
    formatted = []
    for doc, doc_index in zip(retrieved, indices):
        frame_index = cache.get("frame_indices", [])[doc_index] if doc_index < len(cache.get("frame_indices", [])) else -1
        timestamp = cache.get("timestamps", [])[frame_index] if 0 <= frame_index < len(cache.get("timestamps", [])) else None
        prefix = f"[OCR at {timestamp:.2f}s]" if timestamp is not None else "[OCR]"
        formatted.append(f"{prefix} {doc}")
    return {"documents": formatted, "raw_document_count": len(docs), "retrieved_count": len(formatted)}


def unique_new_docs(documents: list[str], baseline_documents: list[str]) -> list[str]:
    baseline_text = "\n".join(str(x).lower() for x in baseline_documents)
    result = []
    for document in documents:
        core = re.sub(r"^\[[^]]+\]\s*", "", str(document)).strip()
        if core and core.lower() not in baseline_text and document not in result:
            result.append(document)
    return result[:16]


def insert_evidence(base_prompt: str, additions: list[str]) -> str:
    additions = [str(x).strip() for x in additions if str(x or "").strip()]
    if not additions:
        return base_prompt
    block = "\n" + "\n".join(additions) + "\n"
    index = base_prompt.find(ANSWER_MARKER)
    return base_prompt[:index] + block + base_prompt[index:] if index >= 0 else base_prompt + block


def merge_chronological_frames(global_frames, global_timestamps: list[float], local_frames, local_timestamps: list[float]):
    entries = []
    for index, (frame, timestamp) in enumerate(zip(global_frames, global_timestamps)):
        entries.append((float(timestamp), 0, index, frame))
    for index, (frame, timestamp) in enumerate(zip(local_frames, local_timestamps)):
        entries.append((float(timestamp), 1, index, frame))
    entries.sort(key=lambda value: (value[0], value[1], value[2]))
    return np.stack([value[3] for value in entries], axis=0) if entries else np.zeros((0, 336, 336, 3), dtype=np.uint8)


def local_metadata(timestamps: list[float], start_index: int = 0, global_frames: int = GLOBAL_FRAMES) -> str:
    mapping = ", ".join(f"L{start_index + i + 1}={ts:.2f}s" for i, ts in enumerate(timestamps))
    return (
        f"Additional question-routed visual keyframes are marked as {mapping}. "
        f"They are merged with the original {global_frames} uniform frames in chronological timestamp order. "
        "The burned-in L labels and timestamps identify only the added keyframes."
    )


def evidence_lines(
    asr_docs: list[str], ocr_docs: list[str], local_ts: list[float], global_frames: int = GLOBAL_FRAMES
) -> list[str]:
    lines = []
    # Keep the answer model causally independent from Qwen's interpretation.
    # Qwen's observed facts remain in diagnostics and determine which raw
    # frames/tools are supplied, but its prose and candidate answer are hidden.
    if asr_docs:
        lines.append("Additional question-routed ASR evidence: " + " | ".join(asr_docs))
    if ocr_docs:
        lines.append("Additional question-routed OCR evidence: " + " | ".join(ocr_docs))
    if local_ts:
        lines.append(local_metadata(local_ts, global_frames=global_frames))
    return lines


def evidence_package(base_diag: dict, additions: list[str]) -> str:
    sections = []
    for name, key in (("BASELINE_ASR", "asr_docs_full"), ("BASELINE_OCR", "ocr_docs_full"), ("BASELINE_DET", "det_docs_full")):
        docs = normalize_list(base_diag.get(key))[:12]
        if docs:
            sections.append(name + ": " + " | ".join(docs))
    sections.extend(additions)
    text = "\n".join(sections)
    return text[:16000] if text else "NO_TEXTUAL_EVIDENCE"


def infer_frames(baseline, prompt: str, frames) -> str:
    tensor = baseline.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    try:
        return baseline.llava_inference(prompt, [tensor])
    finally:
        del tensor
        baseline.torch.cuda.empty_cache()


def critic_prompt(question: str, prediction: str, package: str, evidence_frame_count: int) -> str:
    return f"""You are a counter-evidence critic for video multiple-choice QA.
Independently audit the local answer model. Do not merely agree. Find the strongest
competing option and actively search the supplied evidence for a contradiction.
Use only the evidence package and attached visual evidence frames. Do not use a prior
planner answer because it is intentionally hidden.

Question and options: {question}
Answer under review: {prediction}
Attached visual evidence frames: {evidence_frame_count}. Frames labelled E are
planner-grounding frames; frames labelled L are supplementary local/key frames.
Burned-in timestamps are authoritative.

Evidence package with provenance:
{package}

Return SUPPORTED only if the evidence positively supports the answer and rules out
the strongest competitor. Return REFUTED only with concrete contradictory evidence.
Return INSUFFICIENT when a missing modality prevents a reliable decision.

Return JSON only:
{{
  "verdict": "SUPPORTED/REFUTED/INSUFFICIENT",
  "strongest_competing_option": "A/B/C/D/UNKNOWN",
  "supporting_evidence": ["quoted or paraphrased evidence with provenance"],
  "contradictory_evidence": ["concrete counter-evidence with provenance"],
  "evidence_sufficient": true,
  "next_tool": "ASR/OCR/LOCAL_FRAMES/NONE",
  "next_query": "specific missing fact or empty",
  "reason": "short audit explanation"
}}"""


def critic_request_signature(
    item: dict,
    prediction: str,
    package: str,
    evidence_timestamps: list[float],
    api_base: str,
    api_model: str,
) -> str:
    return stable_hash({
        "protocol": CRITIC_PROTOCOL_VERSION,
        "question": item["question"],
        "video_stem": Path(item["video_path"]).stem,
        "prediction": prediction,
        "evidence_package": package,
        "evidence_timestamps": evidence_timestamps,
        "api_base": api_base,
        "api_model": api_model,
    })


def run_critic(api_key: str, api_base: str, api_model: str, item: dict, prediction: str, package: str, evidence_frames) -> dict:
    try:
        raw = api_chat(
            api_key,
            api_base,
            api_model,
            critic_prompt(item["question"], prediction, package, len(evidence_frames)),
            list(evidence_frames) if len(evidence_frames) else None,
            max_tokens=512,
        )
    except ApiContentBlocked as exc:
        print(f"[Qwen content blocked] critic {item['question_id']}; keeping the pre-reflection answer.")
        return {
            "api_status": "content_blocked",
            "api_error": str(exc),
            "raw_response": "",
            "parsed": {},
            "verdict": "INSUFFICIENT",
            "strongest_competing_option": "",
            "supporting_evidence": [],
            "contradictory_evidence": [],
            "evidence_sufficient": False,
            "next_tool": "NONE",
            "next_query": "",
            "reason": "Qwen rejected the critic input during content inspection.",
        }
    parsed = parse_json_loose(raw)
    verdict = str(parsed.get("verdict") or "INSUFFICIENT").upper()
    if verdict not in {"SUPPORTED", "REFUTED", "INSUFFICIENT"}:
        verdict = "INSUFFICIENT"
    next_tools = normalize_tools([parsed.get("next_tool")])
    return {
        "api_status": "ok",
        "api_error": "",
        "raw_response": raw,
        "parsed": parsed,
        "verdict": verdict,
        "strongest_competing_option": extract_letter(parsed.get("strongest_competing_option")),
        "supporting_evidence": clean_grounded_audit_evidence(parsed.get("supporting_evidence")),
        "contradictory_evidence": clean_grounded_audit_evidence(parsed.get("contradictory_evidence")),
        "evidence_sufficient": parse_bool(parsed.get("evidence_sufficient")),
        "next_tool": next_tools[0] if next_tools else "NONE",
        "next_query": str(parsed.get("next_query") or "").strip(),
        "reason": str(parsed.get("reason") or "").strip(),
    }


def method_result(raw: str, ground_truth: str) -> dict:
    prediction = extract_letter(raw)
    return {"raw_output": raw, "prediction": prediction, "correct": prediction == ground_truth}


def tool_request_signature(item: dict, planner: dict, base_diag: dict) -> str:
    return stable_hash({
        "protocol": TOOL_PROTOCOL_VERSION,
        "question": item["question"],
        "planner_signature": planner.get("request_signature"),
        "tool_requests": planner.get("tool_requests", []),
        "asr_query": planner.get("asr_query", ""),
        "ocr_keywords": planner.get("ocr_keywords", []),
        "baseline_prompt_hash": stable_hash(base_diag.get("final_prompt", "")),
    })


def acquire_tools(baseline, item: dict, planner: dict, base_diag: dict, output_dir: Path) -> dict:
    tools = planner.get("tool_requests", [])
    asr = retrieve_asr(baseline, item, planner.get("asr_query", ""), output_dir) if "ASR" in tools else {"documents": [], "retrieved_count": 0}
    ocr = retrieve_ocr(baseline, item, planner.get("ocr_keywords", []), output_dir) if "OCR" in tools else {"documents": [], "retrieved_count": 0}
    return {
        "request_signature": tool_request_signature(item, planner, base_diag),
        "requested_tools": tools,
        "asr": asr,
        "ocr": ocr,
        "new_asr_documents": unique_new_docs(asr.get("documents", []), base_diag.get("asr_docs_full", [])),
        "new_ocr_documents": unique_new_docs(ocr.get("documents", []), base_diag.get("ocr_docs_full", [])),
    }


def mcnemar_exact(gains: int, losses: int) -> float:
    total = gains + losses
    if total == 0:
        return 1.0
    tail = min(gains, losses)
    return min(1.0, 2 * sum(math.comb(total, k) for k in range(tail + 1)) / (2 ** total))


def summarize(records: dict[str, dict], expected: int) -> dict:
    summary = {"expected_questions": expected, "completed_questions": len(records), "methods": {}, "paired_vs_baseline": {}, "by_level": {}, "routing": {}, "reflection": {}}
    for method in METHODS:
        values = [row["methods"][method] for row in records.values()]
        correct = sum(bool(x["correct"]) for x in values)
        summary["methods"][method] = {"correct": correct, "total": len(values), "accuracy": round(correct / len(values), 4) if values else 0.0}
    for method in ("agent_pre_reflection", "agent_final"):
        gains = losses = 0
        for row in records.values():
            baseline_ok = bool(row["methods"]["baseline_paired"]["correct"])
            method_ok = bool(row["methods"][method]["correct"])
            gains += not baseline_ok and method_ok
            losses += baseline_ok and not method_ok
        summary["paired_vs_baseline"][method] = {"gains": gains, "losses": losses, "net_correct": gains - losses, "mcnemar_exact_p": round(mcnemar_exact(gains, losses), 6)}
    for level in ("A", "B", "C", "UNKNOWN"):
        subset = [row for row in records.values() if row["question_level"] == level]
        summary["by_level"][level] = {}
        for method in METHODS:
            correct = sum(bool(row["methods"][method]["correct"]) for row in subset)
            summary["by_level"][level][method] = {"correct": correct, "total": len(subset), "accuracy": round(correct / len(subset), 4) if subset else 0.0}
    planners = [row["planner"] for row in records.values()]
    summary["routing"] = {
        "planner_api_status_distribution": dict(Counter(x.get("api_status", "ok") for x in planners)),
        "tool_request_distribution": dict(Counter(",".join(x.get("tool_requests", [])) or "NONE" for x in planners)),
        "missing_evidence_distribution": dict(Counter(x.get("missing_evidence", "none") for x in planners)),
        "planner_evidence_sufficient": sum(bool(x.get("evidence_sufficient")) for x in planners),
        "questions_with_local_frames": sum(bool(row.get("local_frame_timestamps")) for row in records.values()),
        "average_final_frame_count": round(sum(row.get("final_frame_count", GLOBAL_FRAMES) for row in records.values()) / len(records), 4) if records else 0.0,
    }
    critics = [row["critic"] for row in records.values()]
    pre_to_final_corrected = sum(not row["methods"]["agent_pre_reflection"]["correct"] and row["methods"]["agent_final"]["correct"] for row in records.values())
    pre_to_final_corrupted = sum(row["methods"]["agent_pre_reflection"]["correct"] and not row["methods"]["agent_final"]["correct"] for row in records.values())
    summary["reflection"] = {
        "critic_api_status_distribution": dict(Counter(x.get("api_status", "ok") for x in critics)),
        "verdict_distribution": dict(Counter(x.get("verdict", "INSUFFICIENT") for x in critics)),
        "reanswer_count": sum(bool(row.get("reflection_reanswer")) for row in records.values()),
        "corrected_errors": pre_to_final_corrected,
        "corrupted_correct_answers": pre_to_final_corrupted,
        "net_corrections": pre_to_final_corrected - pre_to_final_corrupted,
    }
    return summary


def write_outputs(output_dir: Path, records: dict[str, dict], expected: int) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summarize(records, expected), ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["question_id", "question_level", "task_type", "ground_truth"]
    for method in METHODS:
        fields += [f"{method}_prediction", f"{method}_correct"]
    fields += ["tools", "planner_api_status", "critic_api_status", "critic_verdict", "reflection_reanswer", "final_frame_count"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid in sorted(records):
            record = records[qid]
            row = {key: record[key] for key in ("question_id", "question_level", "task_type", "ground_truth")}
            for method in METHODS:
                row[f"{method}_prediction"] = record["methods"][method]["prediction"]
                row[f"{method}_correct"] = record["methods"][method]["correct"]
            row["tools"] = ",".join(record["planner"].get("tool_requests", []))
            row["planner_api_status"] = record["planner"].get("api_status", "ok")
            row["critic_api_status"] = record["critic"].get("api_status", "ok")
            row["critic_verdict"] = record["critic"].get("verdict")
            row["reflection_reanswer"] = record.get("reflection_reanswer")
            row["final_frame_count"] = record.get("final_frame_count")
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Evidence-gap reflective Video-RAG Agent")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--json", default=str(DEFAULT_ANNOTATION))
    parser.add_argument("--level_jsonl", default=str(DEFAULT_LEVELS))
    parser.add_argument("--baseline_seed", default=str(DEFAULT_BASELINE_SEED))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--planner_frames", type=int, default=PLANNER_FRAMES)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--expected_questions", type=int, default=EXPECTED_FULL_QUESTIONS)
    parser.add_argument("--allow_partial_dataset", action="store_true")
    parser.add_argument("--api_key", default=QWEN_API_KEY or os.environ.get("UNIFIED_QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "")
    parser.add_argument("--api_base", default=QWEN_API_BASE)
    parser.add_argument("--api_model", default=QWEN_API_MODEL)
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = build_question_list(Path(args.json).resolve(), Path(args.data_dir).resolve(), Path(args.level_jsonl).resolve())
    if args.max_questions is not None:
        questions = questions[: max(0, args.max_questions)]
    if args.max_questions is None and not args.allow_partial_dataset and len(questions) != args.expected_questions:
        raise RuntimeError(
            f"Full-data guard failed: found {len(questions)} runnable questions, expected {args.expected_questions}. "
            "Copy the complete video folder/annotations to this machine, or use --max_questions N for a smoke test."
        )
    config = {
        "protocol": PROTOCOL_VERSION,
        "planner_protocol": PLANNER_PROTOCOL_VERSION,
        "tool_protocol": TOOL_PROTOCOL_VERSION,
        "critic_protocol": CRITIC_PROTOCOL_VERSION,
        "question_ids_hash": stable_hash([item["question_id"] for item in questions]),
        "annotation_hash": file_hash(Path(args.json).resolve()),
        "level_hash": file_hash(Path(args.level_jsonl).resolve()),
        "baseline_seed_hash": file_hash(Path(args.baseline_seed).resolve()),
        "video_manifest_hash": video_manifest_hash(Path(args.data_dir).resolve()),
        "global_frames": GLOBAL_FRAMES,
        "planner_frames": args.planner_frames,
        "dense_ocr_frames": DENSE_OCR_FRAMES,
        "max_local_frames": MAX_LOCAL_FRAMES,
        "planner_facts_passed_to_answer": False,
        "api_base": args.api_base,
        "api_model": args.api_model,
    }
    config_fingerprint = stable_hash(config)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("config_fingerprint") != config_fingerprint and cache_files_exist(output_dir):
            raise RuntimeError(
                "The output directory contains caches from a different experiment configuration. "
                "Use a new --output_dir so old planner/tool/critic results cannot contaminate this run."
            )
    elif cache_files_exist(output_dir):
        raise RuntimeError("The output directory contains unversioned caches. Use a new --output_dir.")
    manifest = {
        "total_questions": len(questions),
        "expected_questions": args.expected_questions,
        "by_level": dict(Counter(x["question_level"] for x in questions)),
        "config_fingerprint": config_fingerprint,
        "config": config,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Manifest] {manifest}")
    if args.prepare_only:
        return 0
    if not args.api_key:
        raise RuntimeError("Qwen API key is empty. Fill QWEN_API_KEY at the top of this file.")

    # Only one copy of the local models is loaded.
    import video_evidence_backend as baseline

    seed_rows = load_jsonl(Path(args.baseline_seed).resolve())
    generated_baseline_path = output_dir / "generated_baseline_cache.jsonl"
    generated_baseline = load_jsonl(generated_baseline_path)
    planner_path = output_dir / "planner_cache.jsonl"
    planner_cache = load_jsonl(planner_path)
    tool_path = output_dir / "tool_cache.jsonl"
    tool_cache = load_jsonl(tool_path)
    critic_path = output_dir / "critic_cache.jsonl"
    critic_cache = load_jsonl(critic_path)
    results_path = output_dir / "results.jsonl"
    records = load_jsonl(results_path)

    pending = [
        item for item in questions
        if records.get(item["question_id"], {}).get("config_fingerprint") != config_fingerprint
    ]
    for position, item in enumerate(pending, 1):
        qid = item["question_id"]
        print(f"\n[{position}/{len(pending)}] {qid} level={item['question_level']}")
        generated_signature = stable_hash({
            "protocol": PROTOCOL_VERSION,
            "stage": "generated_baseline",
            "question": item["question"],
            "video_stem": Path(item["video_path"]).stem,
            "global_frames": GLOBAL_FRAMES,
        })
        seed = seed_rows.get(qid)
        if seed is not None and (
            seed.get("question") != item["question"]
            or str(seed.get("video_id") or "") != item["video_id"]
        ):
            seed = None
        if seed is None:
            cached_seed = generated_baseline.get(qid)
            seed = cached_seed if cached_seed and cached_seed.get("request_signature") == generated_signature else None
        if seed is None:
            diagnostics = baseline.video_qa(item["video_path"], item["question"], max_frames_num=GLOBAL_FRAMES)
            seed = {
                "question_id": qid,
                "request_signature": generated_signature,
                "diagnostics": diagnostics,
                "prediction": baseline.extract_answer(diagnostics.get("final_answer")),
            }
            append_jsonl(generated_baseline_path, seed)
            generated_baseline[qid] = seed
        base_diag = seed.get("diagnostics") or {}
        base_prompt = str(base_diag.get("final_prompt") or "")
        if not base_prompt:
            raise RuntimeError(f"Baseline prompt missing for {qid}")

        global_frames, global_time_string, _ = baseline.process_video(item["video_path"], GLOBAL_FRAMES, 1, force_sample=True)
        global_ts = [float(x.replace("s", "")) for x in global_time_string.split(",")]
        paired_baseline_raw = infer_frames(baseline, base_prompt, global_frames)

        expected_planner_signature = planner_request_signature(item, args.api_base, args.api_model, args.planner_frames)
        planner = planner_cache.get(qid)
        if planner is None or planner.get("request_signature") != expected_planner_signature:
            planner = run_planner(baseline, item, args.api_key, args.api_base, args.api_model, args.planner_frames)
            append_jsonl(planner_path, planner)
            planner_cache[qid] = planner

        expected_tool_signature = tool_request_signature(item, planner, base_diag)
        tools = tool_cache.get(qid)
        if tools is None or tools.get("request_signature") != expected_tool_signature:
            tools = {"question_id": qid, **acquire_tools(baseline, item, planner, base_diag, output_dir)}
            append_jsonl(tool_path, tools)
            tool_cache[qid] = tools

        local_frames = np.zeros((0, 336, 336, 3), dtype=np.uint8)
        local_ts = []
        if planner.get("keyframe_valid"):
            local_raw, local_ts = sample_unique_frames(item["video_path"], proposed_keyframe_timestamps(planner), global_ts)
            local_frames = stamp_frames(local_raw, local_ts, "L")
        final_frames = merge_chronological_frames(global_frames, global_ts, local_frames, local_ts)
        additions = evidence_lines(tools.get("new_asr_documents", []), tools.get("new_ocr_documents", []), local_ts)
        agent_prompt = insert_evidence(base_prompt, additions)
        pre_raw = infer_frames(baseline, agent_prompt, final_frames)
        pre_prediction = extract_letter(pre_raw)
        package = evidence_package(base_diag, additions)

        planner_frames_for_critic, planner_evidence_ts = planner_evidence_frames(
            baseline, item, planner, args.planner_frames
        )
        critic_frames = merge_chronological_frames(
            planner_frames_for_critic, planner_evidence_ts, local_frames, local_ts
        )
        critic_evidence_ts = planner_evidence_ts + local_ts
        expected_critic_signature = critic_request_signature(
            item, pre_prediction, package, critic_evidence_ts, args.api_base, args.api_model
        )
        critic = critic_cache.get(qid)
        if critic is None or critic.get("request_signature") != expected_critic_signature:
            if planner.get("api_status") == "content_blocked":
                critic = {
                    "question_id": qid,
                    "answer_under_review": pre_prediction,
                    "request_signature": expected_critic_signature,
                    "visual_evidence_timestamps": [],
                    "api_status": "skipped_planner_content_blocked",
                    "api_error": "",
                    "raw_response": "",
                    "parsed": {},
                    "verdict": "INSUFFICIENT",
                    "strongest_competing_option": "",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                    "evidence_sufficient": False,
                    "next_tool": "NONE",
                    "next_query": "",
                    "reason": "Critic skipped because the storyboard planner was blocked by API content inspection.",
                }
            else:
                critic = {
                    "question_id": qid,
                    "answer_under_review": pre_prediction,
                    "request_signature": expected_critic_signature,
                    "visual_evidence_timestamps": critic_evidence_ts,
                    **run_critic(args.api_key, args.api_base, args.api_model, item, pre_prediction, package, critic_frames),
                }
            append_jsonl(critic_path, critic)
            critic_cache[qid] = critic

        reflection_additions = []
        reflection_reanswer = False
        reflection_tool = "NONE"
        # Concrete counter-evidence is allowed to trigger one independent re-answer.
        if (
            critic.get("verdict") == "REFUTED"
            and critic.get("evidence_sufficient")
            and critic.get("contradictory_evidence")
        ):
            reflection_additions.append("Counter-evidence audit: " + " | ".join(critic["contradictory_evidence"]))
        elif critic.get("verdict") in {"REFUTED", "INSUFFICIENT"}:
            reflection_tool = critic.get("next_tool", "NONE")
            next_query = critic.get("next_query", "")
            if reflection_tool == "ASR":
                extra = retrieve_asr(baseline, item, next_query)
                new_docs = unique_new_docs(extra.get("documents", []), base_diag.get("asr_docs_full", []) + tools.get("new_asr_documents", []))
                if new_docs:
                    reflection_additions.append("Reflection-requested ASR evidence: " + " | ".join(new_docs))
            elif reflection_tool == "OCR":
                extra = retrieve_ocr(baseline, item, normalize_list(next_query), output_dir)
                new_docs = unique_new_docs(extra.get("documents", []), base_diag.get("ocr_docs_full", []) + tools.get("new_ocr_documents", []))
                if new_docs:
                    reflection_additions.append("Reflection-requested OCR evidence: " + " | ".join(new_docs))
            elif reflection_tool == "LOCAL_FRAMES" and len(local_ts) < MAX_LOCAL_FRAMES:
                reflection_plan = dict(planner)
                reflection_anchors = list(planner.get("anchor_timestamps", []))
                if not reflection_anchors:
                    candidate_ts = planner.get("candidate_timestamps", [])
                    reflection_anchors = [
                        candidate_ts[index]
                        for index in planner.get("visual_evidence_frame_indices", [])[:3]
                        if 0 <= index < len(candidate_ts)
                    ]
                reflection_plan["anchor_timestamps"] = reflection_anchors
                extra_raw, extra_ts = sample_unique_frames(
                    item["video_path"], proposed_keyframe_timestamps(reflection_plan, force=True), global_ts + local_ts
                )
                remaining = MAX_LOCAL_FRAMES - len(local_ts)
                extra_raw, extra_ts = extra_raw[:remaining], extra_ts[:remaining]
                if extra_ts:
                    prior_count = len(local_ts)
                    extra_frames = stamp_frames(extra_raw, extra_ts, "L", start_index=prior_count)
                    local_frames = np.concatenate([local_frames, extra_frames], axis=0) if len(local_frames) else extra_frames
                    local_ts.extend(extra_ts)
                    final_frames = merge_chronological_frames(global_frames, global_ts, local_frames, local_ts)
                    reflection_additions.append(local_metadata(extra_ts, start_index=prior_count))

        if reflection_additions:
            reflection_reanswer = True
            final_prompt = insert_evidence(agent_prompt, reflection_additions)
            final_raw = infer_frames(baseline, final_prompt, final_frames)
        else:
            final_prompt = agent_prompt
            final_raw = pre_raw

        ground_truth = item["ground_truth"]
        reference_raw = str(base_diag.get("final_answer") or seed.get("raw_output") or seed.get("prediction") or "")
        methods = {
            "baseline_reference": method_result(reference_raw, ground_truth),
            "baseline_paired": method_result(paired_baseline_raw, ground_truth),
            "qwen_planner_candidate": method_result(planner.get("candidate_answer", ""), ground_truth),
            "agent_pre_reflection": method_result(pre_raw, ground_truth),
            "agent_final": method_result(final_raw, ground_truth),
        }
        record = {
            **{key: item[key] for key in ("video_id", "question_id", "question_level", "task_type", "video_category", "video_subcategory", "question", "ground_truth")},
            "config_fingerprint": config_fingerprint,
            "methods": methods,
            "planner": planner,
            "tools": tools,
            "critic": critic,
            "local_frame_timestamps": local_ts,
            "final_frame_count": len(final_frames),
            "reflection_reanswer": reflection_reanswer,
            "reflection_tool": reflection_tool,
            "reflection_additions": reflection_additions,
            "baseline_prompt": base_prompt,
            "agent_prompt": agent_prompt,
            "final_prompt": final_prompt,
        }
        append_jsonl(results_path, record)
        records[qid] = record
        write_outputs(output_dir, records, len(questions))
        print("[Done]", qid, "GT=", ground_truth, {name: data["prediction"] for name, data in methods.items()}, "tools=", planner.get("tool_requests"), "critic=", critic.get("verdict"))

    write_outputs(output_dir, records, len(questions))
    print(f"\n[Complete] {len(records)}/{len(questions)}")
    print(f"[Results] {results_path}")
    print(f"[Summary] {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
