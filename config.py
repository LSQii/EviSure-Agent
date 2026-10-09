"""Edit this file, then run: python main.py --dataset videomme

All relative paths are resolved from this folder.
"""

# Dataset paths. JSON, JSONL, and Video-MME-style nested JSON are supported.
ANNOTATION = "example_dataset.jsonl"
VIDEO_DIR = "data/videos"
OUTPUT_DIR = "outputs/our_method"
DATASET_FORMAT = "auto"  # auto / videomme / flat_json / flat_jsonl

# Formal paper setting: [16]. Use [4], [8], or [4, 8, 16] for frame ablations.
ANSWER_FRAMES = [16]
PLANNER_FRAMES = 32
RELATION_STORYBOARD_FRAMES = 48

# auto: use provided question_level/task_type; call Qwen only when either is missing.
# metadata: never infer missing routing labels.
ROUTING_MODE = "auto"  # auto / metadata

# Fill locally. Environment variables DASHSCOPE_API_KEY and UNIFIED_QWEN_API_KEY
# override this value when they are set.
# Optional local fallback. Prefer the environment variables above.
API_KEY = ""
API_BASE = "https://YOUR_COMPATIBLE_MODE_ENDPOINT/v1"
API_MODEL = "qwen3-vl-plus"

# Optional run controls.
MAX_QUESTIONS = None
ALLOW_MISSING_VIDEOS = False
SAVE_DIAGNOSTICS = True
