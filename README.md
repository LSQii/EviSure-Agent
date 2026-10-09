# EviSure Agent

EviSure Agent performs evidence-aware video question answering with
LLaVA-Video. It builds an initial answer, retrieves missing evidence, verifies
temporal or causal relations, and changes the answer only when the verification
views agree.

Supported datasets:

- Video-MME
- LongVideoBench
- the seven multiple-choice tasks of MLVU

## Installation

```bash
conda create -n evisure python=3.10 -y
conda activate evisure
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

The project loads these model checkpoints automatically when they are not in
the local Hugging Face cache:

- `lmms-lab/LLaVA-Video-7B-Qwen2`
- `openai/clip-vit-large-patch14-336`
- `openai/whisper-large`
- `facebook/contriever`

## API configuration

Set the API key through an environment variable:

```bash
export UNIFIED_QWEN_API_KEY="YOUR_API_KEY"
# or: export DASHSCOPE_API_KEY="YOUR_API_KEY"
```

The endpoint and model can be changed in `config.py` or with `--api_base` and
`--api_model`.

## Run

Validate paths without loading models:

```bash
python main.py --dataset videomme \
  --annotation /path/to/Video-MME.json \
  --video_dir /path/to/videos \
  --output_dir ./outputs/videomme \
  --routing_mode auto \
  --allow_missing_videos \
  --prepare_only
```

Video-MME:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset videomme \
  --annotation /path/to/Video-MME.json \
  --video_dir /path/to/videos \
  --output_dir ./outputs/videomme \
  --answer_frames 16 \
  --routing_mode auto \
  --allow_missing_videos
```

LongVideoBench:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset longvideobench \
  --annotation /path/to/LongVideoBench/lvb_val.json \
  --video_dir /path/to/LongVideoBench/videos \
  --output_dir ./outputs/longvideobench \
  --answer_frames 16 \
  --routing_mode auto \
  --allow_missing_videos
```

MLVU:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset mlvu \
  --json_dir /path/to/MLVU/json \
  --video_dir /path/to/MLVU/video \
  --output_dir ./outputs/mlvu \
  --answer_frames 16 \
  --routing_mode auto \
  --allow_missing_videos
```

Use `python main.py --dataset <name> --help` to view dataset-specific options.
Runs are resumable; repeat the same command to skip completed records, or add
`--retry_errors` to retry failed records.

## Output

```text
outputs/<dataset>/frames_16/
├── manifest.json
├── predictions.jsonl
├── predictions.csv
├── summary.json
└── evidence_cache/
```

The final answer is stored in `final_prediction`. The main diagnostic fields
are `initial_prediction`, `verifier_route`, `verifier_vote`,
`confirmation_vote1`, `confirmation_vote2`, and `accepted_change`.

## Core files

```text
main.py                         unified entry point
scripts/                        dataset runners
video_evidence_backend.py       video, VLM, CLIP, ASR, and OCR backend
evidence_gap_reflective_agent.py
causal_evidence_verifier.py
post_answer_causal_verifier.py
post_answer_stability_gate.py
llava/                          local LLaVA runtime
tools/                          retrieval utilities
```

This repository includes code derived from LLaVA-NeXT/LLaVA-Video and retains
the Apache License 2.0. Third-party models and datasets use their own licenses.
