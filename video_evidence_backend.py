from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, CLIPProcessor, CLIPModel
import copy
from decord import VideoReader, cpu
import numpy as np
import json
from tqdm import tqdm
import os
import easyocr
from tools.rag_retriever_dynamic import retrieve_documents_with_dynamic
import re
import ast
import socket
import pickle
from tools.filter_keywords import filter_keywords
from tools.scene_graph import generate_scene_graph_description
import ffmpeg, torchaudio
import argparse
import csv
from typing import List, Any
from pathlib import Path

# ==================== 全局配置与模型加载 ====================
max_frames_num = 16

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.float16, device_map="auto")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large",
    torch_dtype=torch.float16,
    device_map="auto"
)
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")

# load your VLM
device = "cuda"
overwrite_config = {}
tokenizer, model, image_processor, max_length = load_pretrained_model(
    "lmms-lab/LLaVA-Video-7B-Qwen2",
    None,
    "llava_qwen",
    torch_dtype="bfloat16",
    device_map="auto",
    overwrite_config=overwrite_config
)
model.eval()
conv_template = "qwen_1_5"

# super-parameters setting
rag_threshold = 0.3
clip_threshold = 0.3
beta = 3.0

# Choose the auxiliary texts you want
USE_OCR = True
USE_ASR = True
USE_DET = True
print(f"---------------OCR(th={rag_threshold}): {USE_OCR}-----------------")
print(f"---------------ASR(th={rag_threshold}): {USE_ASR}-----------------")
print(f"---------------DET(beta={beta}, th={clip_threshold}): {USE_DET}-----------------")
print(f"---------------Frames: {max_frames_num}-----------------")

# EasyOCR 全局单例 —— 避免 batch 中每个 question 都重新加载模型
_easyocr_reader = None

def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        print('[EasyOCR] Initializing reader (one-time)...')
        _easyocr_reader = easyocr.Reader(['en'])
        print('[EasyOCR] Reader initialized.')
    return _easyocr_reader

# ==================== 辅助函数 ====================

def process_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "0.00s", 0.0
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    try:
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
    finally:
        del vr
    return spare_frames, frame_time, video_time


def extract_audio(video_path, audio_path):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    if not os.path.exists(audio_path):
        ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run()


def chunk_audio(audio_path, chunk_length_s=30):
    speech, sr = torchaudio.load(audio_path)
    speech = speech.mean(dim=0)
    speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
    num_samples_per_chunk = chunk_length_s * 16000
    chunks = []
    for i in range(0, len(speech), num_samples_per_chunk):
        chunks.append(speech[i:i + num_samples_per_chunk])
    return chunks


def transcribe_chunk(chunk):
    inputs = whisper_processor(chunk, return_tensors="pt")
    inputs["input_features"] = inputs["input_features"].to(whisper_model.device, torch.float16)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            inputs["input_features"],
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


# asr 语音识别 whisper
def get_asr_docs(video_path, audio_path):
    full_transcription = []
    try:
        extract_audio(video_path, audio_path)
    except Exception as e:
        print(f'[ASR] Audio extraction failed: {e}')
        return full_transcription
    audio_chunks = chunk_audio(audio_path, chunk_length_s=30)
    for chunk in audio_chunks:
        transcription = transcribe_chunk(chunk)
        full_transcription.append(transcription)
    return full_transcription


# ocr 字幕识别  easyOCR
def get_ocr_docs(frames):
    reader = _get_easyocr_reader()
    text_set = []
    ocr_docs = []
    ocr_frame_indices = []          # 记录每段 OCR 文字来自哪一帧
    for frame_idx, img in enumerate(frames):
        ocr_results = reader.readtext(img)
        det_info = ""
        for result in ocr_results:
            text = result[1]
            confidence = result[2]
            if confidence > 0.5 and text not in text_set:
                det_info += f"{text}; "
                text_set.append(text)
        if len(det_info) > 0:
            ocr_docs.append(det_info)
            ocr_frame_indices.append(frame_idx)
    return ocr_docs, ocr_frame_indices


def save_frames(frames, output_dir='restore'):
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        file_path = os.path.join(output_dir, f'frame_{i}.png')
        abs_fp = os.path.abspath(file_path)          # 转为绝对路径
        img.save(abs_fp)
        file_paths.append(abs_fp)
    return file_paths


def get_det_docs(frames: List[Any], prompt: List[str]) -> List:
    """
    向 APE 目标检测服务发送帧路径和文本 prompt，接收检测结果。
    协议：sendall → shutdown(SHUT_WR) → recv_all → close（每个连接一次请求-响应）。

    Args:
        frames: 视频帧列表（numpy 数组）
        prompt: 待检测的目标描述列表，如 ["blue balloons", "long table"]

    Returns:
        检测结果列表，每个元素对应一帧的检测字符串（如 "balloon:(0.1,0.2,0.3,0.4);..."）
        如果连接失败、超时或服务返回错误，返回空列表。
    """
    DET_HOST = '127.0.0.1'
    DET_PORT = 27861
    DET_CONNECT_TIMEOUT = 10
    DET_RECV_TIMEOUT    = 300   # 推理等待超时

    print(f"[DET] Input frames count: {len(frames) if frames is not None else 0}")
    if frames is None or len(frames) == 0:
        print("[DET] No frames provided, returning empty list")
        return []

    prompt = ','.join(prompt)
    print(f"[DET] Prompt: {prompt}")

    out_dir = 'restore/det_frames'
    os.makedirs(out_dir, exist_ok=True)
    frames_path = save_frames(frames, output_dir=out_dir)

    res = []
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(DET_CONNECT_TIMEOUT)
        sock.connect((DET_HOST, DET_PORT))
        print(f"[DET] Connected to {DET_HOST}:{DET_PORT}")

        # 发送请求，然后关闭写端表示发送完毕
        sock.sendall(pickle.dumps((frames_path, prompt)))
        sock.shutdown(socket.SHUT_WR)

        # 切换到长超时等待推理结果
        sock.settimeout(DET_RECV_TIMEOUT)
        data = b''
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk

        if data:
            res = pickle.loads(data)
            # 服务端可能返回错误标记 {"error": ...} 而非正常检测结果
            if isinstance(res, dict) and "error" in res:
                print(f"[DET] Service error: {res['error']}")
                res = []
            else:
                print(f"[DET] Result: {len(res) if isinstance(res, list) else 0} detections")
        else:
            print("[DET] Empty response from service")

    except socket.timeout:
        print("[DET] Timeout — detection service did not respond in time")
    except ConnectionRefusedError:
        print(f"[DET] Connection refused — is detection service running on {DET_HOST}:{DET_PORT}?")
    except Exception as e:
        print(f"[DET] Error: {type(e).__name__}: {e}")
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    return res


def det_preprocess(det_docs, location, relation, number):
    scene_descriptions = []
    for det_doc_per_frame in det_docs:
        objects = []
        scene_description = ""
        if len(det_doc_per_frame) > 0:
            for obj_id, objs in enumerate(det_doc_per_frame.split(";")):
                parts = objs.split(":")
                if len(parts) < 2:
                    continue
                obj_name = parts[0].strip()
                obj_bbox = parts[1].strip()
                obj_bbox = ast.literal_eval(obj_bbox)
                objects.append({"id": obj_id, "label": obj_name, "bbox": obj_bbox})
            scene_description = generate_scene_graph_description(objects, location, relation, number)
        scene_descriptions.append(scene_description)
    return scene_descriptions


# The inference function of your VLM
def llava_inference(qs, video):
    if video is not None:
        question = DEFAULT_IMAGE_TOKEN + qs
    else:
        question = qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    # 显式释放 GPU 张量，避免 batch 模式下的显存积累
    del input_ids, cont
    return text_outputs


def clean_json_output(text):
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()


# ==================== 核心封装函数 ====================

def video_qa(video_path, question, max_frames_num=16):
    """
    对单个视频+问题执行完整的 Video-RAG 推理流程。
    保留原始的 prompt 格式（ASR/DET/TYPE 字段），不做修改。
    """

    # step 0: get cot information （保留原始 prompt，不做任何改动）
    retrieve_pmt_0 = "Question: " + question
    # you can change this decouple prompt to fit your requirements
    retrieve_pmt_0 += "\nTo answer the question step by step, you can provide your retrieve request to assist you by the following json format:"
    retrieve_pmt_0 += '''{
    "ASR": Optional[str]. The subtitles of the video that may relavent to the question you want to retrieve, in two sentences. If you no need for this information, please return null.
    "DET": Optional[list]. (The output must include only physical entities, not abstract concepts, less than five entities) All the physical entities and their location related to the question you want to retrieve, not abstract concepts. If you no need for this information, please return null.
    "TYPE": Optional[list]. (The output must be specified as null or a list containing only one or more of the following strings: 'location', 'number', 'relation'. No other values are valid for this field) The information you want to obtain about the detected objects. If you need the object location in the video frame, output "location"; if you need the number of specific object, output "number"; if you need the positional relationship between objects, output "relation".
}
## Example 1:
Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
Your retrieve can be:
{
    "ASR": "The location and the color of balloons, the number of the blue balloons.",
    "DET": ["blue ballons", "long table"],
    "TYPE": ["relation", "number"]
}
## Example 2:
Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
Your retrieve can be:
{
    "ASR": null,
    "DET": ["the man in black", "woman"],
    "TYPE": ["location", "relation"]
}
## Example 3:
Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
Your retrieve can be:
{
    "ASR": "The country recognized worldwide for its comedy.",
    "DET": null,
    "TYPE": null
}
Note that you don't need to answer the question in this step, so you don't need any infomation about the video of image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the infomation you want. Please provide the json format.'''

    json_request = llava_inference(retrieve_pmt_0, None)

    # ===== 解析 JSON，提取各字段 =====
    try:
        req_json = json.loads(clean_json_output(json_request))
    except Exception as e:
        print(f"Failed to parse JSON request: {e}")
        req_json = {}

    # 提取原始 prompt 格式中的字段
    request_asr = req_json.get("ASR")
    request_det = req_json.get("DET")
    request_det_info = req_json.get("TYPE")   # 原始字段名 TYPE

    # 判断各工具是否被模型显式请求（保留全局开关 USE_ASR / USE_OCR / USE_DET）
    use_tool_asr = USE_ASR and (request_asr is not None)
    use_tool_det = USE_DET and (request_det is not None)
    use_tool_ocr = USE_OCR   # 原始设计中 OCR 总是执行（无独立 OCR 字段）

    # ===== DEBUG: 检查 step0 工具选择决策 =====
    print("\n" + "="*60)
    print("[DEBUG-STEP0] LLaVA Tool Selection Decision")
    print("="*60)
    print(f"  Raw JSON response    : {json_request}")
    print(f"  ASR requested        : {request_asr is not None}")
    print(f"  DET requested        : {request_det is not None}")
    print(f"  DET info requested   : {request_det_info}")
    print(f"  Actual flags -> ASR:{use_tool_asr}  OCR:{use_tool_ocr}  DET:{use_tool_det}")
    print("="*60 + "\n")

    # step 1: get docs information
    # 统一取帧流程（均匀采样）
    final_frames, frame_time_str, video_time = process_video(video_path, max_frames_num, 1, force_sample=True)
    raw_video = [f for f in final_frames]

    final_video_tensor = image_processor.preprocess(final_frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    final_video = [final_video_tensor]   # 包装为列表

    query = [question]

    # --- DET 预处理 ---
    if use_tool_det:
        # 构建视频张量（供 CLIP 使用）
        video_tensor = []
        for frame in raw_video:
            processed = clip_processor(images=frame, return_tensors="pt")["pixel_values"].to(clip_model.device, dtype=torch.float16)
            video_tensor.append(processed.squeeze(0))
        video_tensor = torch.stack(video_tensor, dim=0)

    # --- OCR 预处理 ---
    if use_tool_ocr:
        ocr_docs_total, ocr_frame_indices = get_ocr_docs(final_frames)
    else:
        ocr_docs_total, ocr_frame_indices = [], []

    # ===== DEBUG: 检查 OCR 结果 =====
    print("\n" + "="*60)
    print("[DEBUG-OCR] OCR Results (before RAG)")
    print("="*60)
    print(f"  OCR enabled     : {use_tool_ocr}")
    print(f"  OCR docs count  : {len(ocr_docs_total)}")
    if ocr_docs_total:
        print(f"  OCR total chars : {sum(len(d) for d in ocr_docs_total)}")
        for i, doc in enumerate(ocr_docs_total[:3]):
            print(f"  OCR[{i}]: {doc[:150]}")
    else:
        print(f"  [WARNING] OCR detected no text!")
    print("="*60 + "\n")

    # --- ASR 预处理 ---
    if use_tool_asr:
        # Profiling runs may point this at an isolated cache so cold/warm
        # measurements never consume or overwrite the formal experiment cache.
        audio_cache_dir = os.environ.get("VIDRAG_AUDIO_CACHE_DIR", "restore/audio")
        os.makedirs(audio_cache_dir, exist_ok=True)
        audio_cache_path = os.path.join(audio_cache_dir, os.path.basename(video_path).split(".")[0] + ".txt")
        if os.path.exists(audio_cache_path):
            with open(audio_cache_path, 'r', encoding='utf-8') as f:
                asr_docs_total = f.readlines()
        else:
            audio_path = os.path.join(audio_cache_dir, os.path.basename(video_path).split(".")[0] + ".wav")
            asr_docs_total = get_asr_docs(video_path, audio_path)
            os.makedirs(os.path.dirname(audio_cache_path), exist_ok=True)
            with open(audio_cache_path, 'w', encoding='utf-8') as f:
                for doc in asr_docs_total:
                    f.write(doc + '\n')
            # 文本已保存到缓存，删除中间的 WAV 文件以节省磁盘空间（删除缓存）
            try:
                os.remove(audio_path)
                print(f"[ASR] Removed temporary WAV: {audio_path}")
            except Exception as e:
                print(f"[ASR] Warning: Failed to remove temporary WAV {audio_path}: {e}")
    else:
        asr_docs_total = []

    # ===== DEBUG: 检查 ASR 结果 =====
    print("\n" + "="*60)
    print("[DEBUG-ASR] ASR Results (before RAG)")
    print("="*60)
    print(f"  ASR enabled     : {use_tool_asr}")
    print(f"  ASR docs count  : {len(asr_docs_total)}")
    if asr_docs_total:
        print(f"  ASR total chars : {sum(len(d) for d in asr_docs_total)}")
        for i, doc in enumerate(asr_docs_total[:3]):
            print(f"  ASR chunk[{i}]: {doc[:200]}")
    else:
        print(f"  [WARNING] No ASR transcripts obtained!")
    print("="*60 + "\n")

    # ==================== 步骤 2：按需进行 RAG 检索 ====================

    # DET 检索（APE fetch）
    det_docs = []
    det_top_idx = []
    if use_tool_det:
        try:
            print("the det need of", request_det)
            request_det_clean = filter_keywords(request_det) if request_det else None
            clip_text = ["A picture of " + txt for txt in request_det_clean] if request_det_clean else []
            if len(clip_text) == 0:
                clip_text = ["A picture of object"]
        except:
            request_det_clean = None
            clip_text = ["A picture of object"]

        clip_inputs = clip_processor(text=clip_text, return_tensors="pt", padding=True, truncation=True).to(clip_model.device)
        clip_img_feats = clip_model.get_image_features(video_tensor)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**clip_inputs)
            similarities = (clip_img_feats @ text_features.T).squeeze(0).mean(1).cpu()
            similarities = np.array(similarities, dtype=np.float64)
            alpha = beta * (len(similarities) / 16)
            similarities = similarities * alpha / np.sum(similarities)

        print("the corresond of image", similarities)
        del clip_inputs, clip_img_feats, text_features
        torch.cuda.empty_cache()

        num_frames = len(final_frames)   # 动态获取帧数
        print("总的帧数为", num_frames)
        det_top_idx = [idx for idx in range(num_frames) if similarities[idx] > clip_threshold]
        print("the request_det_clean", request_det_clean)
        print("det_top_idx", det_top_idx)

        # ===== DEBUG: 检查 DET CLIP 帧选择 =====
        print("\n" + "="*60)
        print("[DEBUG-DET-CLIP] CLIP Frame Selection")
        print("="*60)
        print(f"  Total frames        : {num_frames}")
        print(f"  CLIP text prompts   : {clip_text}")
        print(f"  CLIP threshold      : {clip_threshold}")
        print(f"  Similarities        : {[round(float(s), 4) for s in similarities]}")
        print(f"  Selected frame idx  : {det_top_idx}")
        print(f"  Selected/Total      : {len(det_top_idx)}/{num_frames}")
        if len(det_top_idx) == 0:
            print(f"  [WARNING] No frames passed CLIP threshold!")
        print("="*60 + "\n")

        if request_det_clean is not None and len(request_det_clean) > 0 and len(det_top_idx) > 0:
            det_docs = get_det_docs(final_frames[det_top_idx], request_det_clean)
            print("the det_doc的对应结果为", det_docs)

            L, R, N = False, False, False
            if request_det_info is not None:
                if "location" in request_det_info:
                    L = True
                if "relation" in request_det_info:
                    R = True
                if "number" in request_det_info:
                    N = True
            det_docs = det_preprocess(det_docs, location=L, relation=R, number=N)  # pre-process of APE information

            # ===== DEBUG: 检查 DET 预处理后结果 =====
            print("\n" + "="*60)
            print("[DEBUG-DET-SERVICE] DET Results After Preprocess")
            print("="*60)
            print(f"  Det docs count      : {len(det_docs)}")
            if det_docs:
                for i, doc in enumerate(det_docs):
                    print(f"  Det doc[{i}]: {str(doc)[:300]}")
            else:
                print(f"  [WARNING] DET returned empty after preprocessing!")
            print("="*60 + "\n")

    # OCR 检索
    ocr_docs = []
    ocr_rag_indices = []   # RAG 返回的在 ocr_docs_total 中的索引
    if use_tool_ocr and len(ocr_docs_total) > 0:
        ocr_query = query.copy()
        if request_det is not None and len(request_det) > 0:
            ocr_query.extend(filter_keywords(request_det))
        ocr_docs, ocr_rag_indices = retrieve_documents_with_dynamic(ocr_docs_total, ocr_query, threshold=rag_threshold)

    # ASR 检索
    asr_docs = []
    if use_tool_asr and len(asr_docs_total) > 0:
        asr_query = query.copy()
        if request_asr is not None:
            asr_query.append(request_asr)
        asr_docs, _ = retrieve_documents_with_dynamic(asr_docs_total, asr_query, threshold=rag_threshold)

    # ===== DEBUG: 检查 RAG 检索结果 =====
    print("\n" + "="*60)
    print("[DEBUG-RAG] RAG Retrieval Results (threshold={})".format(rag_threshold))
    print("="*60)
    print(f"  [ASR RAG] total:{len(asr_docs_total)} -> retrieved:{len(asr_docs)}")
    if asr_docs:
        for i, doc in enumerate(asr_docs[:2]):
            print(f"    ASR doc[{i}]: {doc[:200]}")
    print(f"  [OCR RAG] total:{len(ocr_docs_total)} -> retrieved:{len(ocr_docs)}")
    if ocr_docs:
        for i, doc in enumerate(ocr_docs[:2]):
            print(f"    OCR doc[{i}]: {doc[:200]}")
    print(f"  [DET]     det_docs count: {len(det_docs)}")
    print("="*60 + "\n")

    # ==================== 步骤 3：拼接辅助信息 ====================
    qs = ""
    if use_tool_det and len(det_docs) > 0:
        for i, info in enumerate(det_docs):
            if len(info) > 0:
                qs += f"Frame {str(det_top_idx[i]+1)}: " + info + "\n"
        if len(qs) > 0:
            qs = f"\nVideo have {str(max_frames_num)} frames in total, the detected objects' information in specific frames: " + qs
    if use_tool_asr and len(asr_docs) > 0:
        qs += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join(asr_docs)
    if use_tool_ocr and len(ocr_docs) > 0:
        qs += "\nVideo OCR information (given in chronological order of the video): " + "; ".join(ocr_docs)
    # 保留原始 prompt 的最终指令
    option_letters = list(dict.fromkeys(
        value.upper()
        for value in re.findall(
            r"(?:^|\n)\s*([A-Q])[\.\)\:]\s+",
            str(question),
            flags=re.I,
        )
    ))
    if not option_letters:
        option_letters = list("ABCD")
    if len(option_letters) == 1:
        option_text = option_letters[0]
    elif len(option_letters) == 2:
        option_text = " or ".join(option_letters)
    else:
        option_text = ", ".join(option_letters[:-1]) + ", or " + option_letters[-1]
    qs += (
        "Select the best answer to the following multiple-choice question based on "
        "the video and the information (if given). Respond with only the letter "
        f"({option_text}) of the correct option. Question: " + question
    )

    # ===== DEBUG: 检查最终 Prompt =====
    print("\n" + "="*60)
    print("[DEBUG-FINAL-PROMPT] Final Prompt to LLaVA")
    print("="*60)
    print(f"  Final frames : {len(final_frames)}")
    print(f"  Prompt chars : {len(qs)}")
    print(f"  Prompt       :\n{qs}")
    print("="*60 + "\n")

    # ==================== 最终推理 ====================
    res = llava_inference(qs, final_video)
    print(res)
    final_frame_count = len(final_frames)

    # ==================== 显存/内存清理 ====================
    # 释放 final_video GPU 张量
    del final_video, final_video_tensor
    # 释放 DET 相关的 CLIP video_tensor（如果存在）
    try:
        del video_tensor
    except NameError:
        pass
    # 清理 large CPU-side numpy 帧数组，帮助 GC
    try:
        del raw_video
    except NameError:
        pass
    try:
        del final_frames
    except NameError:
        pass
    # 清空 PyTorch CUDA 缓存，将闲置显存归还给驱动
    torch.cuda.empty_cache()

    # 返回对应的判断结果
    diagnostics = {
        "asr_requested": request_asr is not None,
        "det_requested": request_det is not None,
        "det_info": request_det_info,
        "asr_docs_count": len(asr_docs),
        "ocr_docs_count": len(ocr_docs),
        "det_docs_count": len(det_docs),
        "asr_docs_preview": [str(doc)[:1000] for doc in asr_docs[:5]],
        "ocr_docs_preview": [str(doc)[:1000] for doc in ocr_docs[:5]],
        "det_docs_preview": [str(doc)[:1000] for doc in det_docs[:5]],
        "asr_docs_full": asr_docs,
        "ocr_docs_full": ocr_docs,
        "det_docs_full": det_docs,
        "final_prompt": qs,
        "final_prompt_chars": len(qs),
        "final_frame_count": final_frame_count,
        "frame_time_str": frame_time_str,
        "final_answer": res
    }

    print(f'[Answer] {res}')

    # ===== DEBUG: 检查最终答案与 diagnostics =====
    print("\n" + "="*60)
    print("[DEBUG-FINAL-ANSWER] Summary")
    print("="*60)
    print(f"  Raw output    : {res}")
    print(f"  diagnostics   : {json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str)}")
    print("="*60 + "\n")

    return diagnostics   # 返回字典，包含最终答案和各工具诊断信息


# ==================== 批量评估模块 ====================

def extract_answer(raw_text):
    if not raw_text:
        return ""
    text = raw_text.strip().upper()
    # 1. 优先匹配显式声明：Answer: B / The answer is B / Option B
    m = re.search(r'ANSWER\s*[:IS]\s*([A-D])', text)
    if m:
        return m.group(1)
    # 2. 匹配 "B."、"B)"、"(B)"、"Option B" 等常见格式
    m = re.search(r'\b([A-D])[.\)\s]', text)
    if m:
        return m.group(1)
    # 3. 匹配独立单词边界的字母
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    # 4. 兜底：如果全文只出现唯一一个 A-D 字母，就认它
    letters = [c for c in text if c in 'ABCD']
    if len(set(letters)) == 1:
        return letters[0]
    return ""


def batch_evaluate(data_dir, json_path, output_dir='results', max_frames_num=16):
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    video_map = {}
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        for fp in Path(data_dir).glob(ext):
            video_map[fp.stem] = str(fp)

    print(f'[Batch] Found {len(video_map)} videos in "{data_dir}"')
    print(f'[Batch] Loaded {len(dataset)} entries from "{json_path}"')

    total_correct = total_count = 0
    task_stats = {}
    category_stats = {}

    # 输出文件不使用时间戳后缀（保留固定文件名）
    csv_path = os.path.join(output_dir, 'eval_details.csv')
    jsonl_path = os.path.join(output_dir, 'eval_details.jsonl')
    fieldnames = ['video_id', 'question_id', 'task_type', 'video_category',
                'video_subcategory', 'question', 'ground_truth',
                'prediction', 'raw_output', 'correct']

    # 写入 CSV header，同时打开 JSONL 用于增量写入（不往内存里积累全量结果）
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csv_f, \
         open(jsonl_path, 'w', encoding='utf-8') as jsonl_f:      # open once, flush after each write
        csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        csv_writer.writeheader()

        # 仅保留错误案例的摘要（用于报告），不保留全量 diagnostics
        error_cases = []   # 十分轻量，只存: question_id, gt, pred, 问题前60字

        for video_item in dataset:
            video_id = video_item.get('video_id', 'unknown')
            url = video_item.get('url', '')
            video_category = video_item.get('video_category', 'Unknown')

            youtube_id = None
            if 'watch?v=' in url:
                youtube_id = url.split('watch?v=')[-1].split('&')[0]
            elif 'youtu.be/' in url:
                youtube_id = url.split('youtu.be/')[-1].split('?')[0]

            if not youtube_id or youtube_id not in video_map:
                print(f'[Skip] {video_id} (yt_id={youtube_id}) not found in {data_dir}')
                continue

            video_path = video_map[youtube_id]
            questions = video_item.get('questions', [])
            print(f'\n[Processing] {video_id} | {youtube_id} | {len(questions)} questions')

            for q_item in questions:
                q_id = q_item.get('question_id', 'unknown')
                task_type = q_item.get('task_type', 'Unknown')
                question_text = q_item.get('question', '')
                choices = q_item.get('choices', [])
                ground_truth = q_item.get('answer', '').strip().upper()

                full_question = question_text + ' ' + ' '.join(choices)

                try:
                    diag = video_qa(video_path, full_question, max_frames_num=max_frames_num)
                    raw_pred = diag.get("final_answer", "")          # 提取最终答案文本
                    pred_letter = extract_answer(raw_pred)           # 提取选项字母
                except Exception as e:
                    print(f'[Error] {q_id}: {e}')
                    raw_pred, pred_letter = '', 'ERROR'
                    diag = {"error": str(e)}                         # 异常时也保存错误信息
                    raise   # 重新抛出异常，程序终止

                is_correct = (pred_letter == ground_truth) if ground_truth else False
                total_count += 1
                if is_correct:
                    total_correct += 1

                for stat_dict, key in [(task_stats, task_type), (category_stats, video_category)]:
                    if key not in stat_dict:
                        stat_dict[key] = {'correct': 0, 'total': 0}
                    stat_dict[key]['total'] += 1
                    if is_correct:
                        stat_dict[key]['correct'] += 1

                # 构建完整结果记录，**立即写入 JSONL 并释放，不留在内存**
                record = {
                    'video_id': video_id,
                    'question_id': q_id,
                    'task_type': task_type,
                    'video_category': video_category,
                    'video_subcategory': video_item.get('video_subcategory', 'Unknown'),
                    'question': full_question,
                    'ground_truth': ground_truth,
                    'prediction': pred_letter,
                    'raw_output': raw_pred,
                    'correct': is_correct,
                    'diagnostics': diag
                }
                # 增量写 JSONL（每行一个 JSON 对象）
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                jsonl_f.flush()
                # 释放 large record dict（diagnostics 可能很大）
                del record, diag

                print(f'  {q_id} [{task_type}] GT={ground_truth} Pred={pred_letter} {"✓" if is_correct else "✗"}')

                # 增量写 CSV
                csv_writer.writerow({
                    'video_id': video_id,
                    'question_id': q_id,
                    'task_type': task_type,
                    'video_category': video_category,
                    'video_subcategory': video_item.get('video_subcategory', 'Unknown'),
                    'question': full_question,
                    'ground_truth': ground_truth,
                    'prediction': pred_letter,
                    'raw_output': raw_pred,
                    'correct': is_correct,
                })
                csv_f.flush()

                # 仅记录错误案例的轻量摘要
                if not is_correct:
                    error_cases.append({
                        'question_id': q_id,
                        'ground_truth': ground_truth,
                        'prediction': pred_letter,
                        'question_preview': full_question[:60]
                    })

    overall_acc = total_correct / total_count if total_count > 0 else 0.0

    summary = {
        'overall': {'correct': total_correct, 'total': total_count, 'accuracy': round(overall_acc, 4)},
        'by_task_type': {
            k: {'correct': v['correct'], 'total': v['total'], 'accuracy': round(v['correct']/v['total'], 4)}
            for k, v in task_stats.items()
        },
        'by_category': {
            k: {'correct': v['correct'], 'total': v['total'], 'accuracy': round(v['correct']/v['total'], 4)}
            for k, v in category_stats.items()
        }
    }

    # 写入摘要 JSON（仅 stats，不含全量 results；完整结果在 JSONL 中）
    summary_path = os.path.join(output_dir, 'eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_path = os.path.join(output_dir, 'eval_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('Video QA Batch Evaluation Report\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Overall Accuracy: {overall_acc*100:.2f}% ({total_correct}/{total_count})\n\n')

        f.write('--- Accuracy by Task Type ---\n')
        for task in sorted(task_stats.keys()):
            st = task_stats[task]
            f.write(f'  {task:25s}: {st["correct"]/st["total"]*100:.2f}% ({st["correct"]}/{st["total"]})\n')

        f.write('\n--- Accuracy by Category ---\n')
        for cat in sorted(category_stats.keys()):
            st = category_stats[cat]
            f.write(f'  {cat:25s}: {st["correct"]/st["total"]*100:.2f}% ({st["correct"]}/{st["total"]})\n')

        f.write('\n--- Error Cases ---\n')
        for r in error_cases:
            f.write(f'  {r["question_id"]}: GT={r["ground_truth"]} Pred={r["prediction"]} | {r["question_preview"]}...\n')

    print(f'\n{"="*60}')
    print(f'Batch Evaluation Complete! Overall: {overall_acc*100:.2f}% ({total_correct}/{total_count})')
    print(f'  JSONL  : {jsonl_path}')
    print(f'  Summary: {summary_path}')
    print(f'  CSV    : {csv_path}')
    print(f'  TXT    : {report_path}')
    print(f'{"="*60}')
    return {'summary': summary, 'paths': {'jsonl': jsonl_path, 'summary': summary_path, 'csv': csv_path, 'txt': report_path}}


def query_video_qa_by_id(
    json_path: str,
    data_dir: str,
    video_id: str,
    question_id: str,
    max_frames_num: int = 16
) -> dict:
    """
    根据 video_id 和 question_id，从标注文件中找到对应问题和视频，
    调用 video_qa 并返回预测结果。同时会在控制台打印类似批量评估的单条结果。
    """
    # 1. 加载标注
    if not os.path.exists(json_path):
        print(f'[Error] JSON file not found: {json_path}')
        return {'error': f'JSON file not found: {json_path}'}
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 2. 构建 video_map
    video_map = {}
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        for fp in Path(data_dir).glob(ext):
            video_map[fp.stem] = str(fp)

    # 3. 查找视频条目
    target_video_item = None
    for item in dataset:
        if item.get('video_id') == video_id:
            target_video_item = item
            break
    if target_video_item is None:
        msg = f'video_id "{video_id}" not found in dataset'
        print(f'[Error] {msg}')
        return {'error': msg}

    # 4. 获取视频路径
    url = target_video_item.get('url', '')
    youtube_id = None
    if 'watch?v=' in url:
        youtube_id = url.split('watch?v=')[-1].split('&')[0]
    elif 'youtu.be/' in url:
        youtube_id = url.split('youtu.be/')[-1].split('?')[0]

    if not youtube_id or youtube_id not in video_map:
        msg = f'Video for {video_id} (yt_id={youtube_id}) not found in {data_dir}'
        print(f'[Error] {msg}')
        return {'error': msg}
    video_path = video_map[youtube_id]

    # 5. 查找问题
    questions = target_video_item.get('questions', [])
    target_question = None
    for q in questions:
        if q.get('question_id') == question_id:
            target_question = q
            break
    if target_question is None:
        msg = f'question_id "{question_id}" not found for video {video_id}'
        print(f'[Error] {msg}')
        return {'error': msg}

    question_text = target_question.get('question', '')
    choices = target_question.get('choices', [])
    ground_truth = target_question.get('answer', '').strip().upper()
    task_type = target_question.get('task_type', 'Unknown')
    full_question = question_text + ' ' + ' '.join(choices)

    # 6. 调用 video_qa
    try:
        diag = video_qa(video_path, full_question, max_frames_num=max_frames_num)
        raw_pred = diag.get("final_answer", "")          # 提取最终答案文本
        pred_letter = extract_answer(raw_pred)           # 提取选项字母
    except Exception as e:
        print(f'[Error] video_qa failed for {question_id}: {e}')
        raw_pred, pred_letter = '', 'ERROR'
        diag = {"error": str(e)}                         # 异常时也保存错误信息
        raise   # 重新抛出异常，程序终止

    print(full_question)
    # 7. 计算正确性并打印（模拟 batch_evaluate 的单条输出）
    is_correct = (pred_letter == ground_truth) if ground_truth else False
    print(f'  {question_id} [{task_type}] GT={ground_truth} Pred={pred_letter} {"✓" if is_correct else "✗"}')

    # 8. 返回完整结果
    return {
        'video_id': video_id,
        'question_id': question_id,
        'task_type': task_type,
        'question': question_text,
        'ground_truth': ground_truth,
        'prediction_raw': raw_pred,
        'prediction': pred_letter,
        'correct': is_correct,
        'video_path': video_path
    }


# ==================== Main 函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Video QA - Single or Batch Evaluation')
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')

    single_parser = subparsers.add_parser('single', help='评估单个视频+问题')
    single_parser.add_argument('--video', type=str, required=True)
    single_parser.add_argument('--question', type=str, required=True)
    single_parser.add_argument('--max_frames', type=int, default=16)
    single_parser.add_argument('--output', type=str, default=None)

    batch_parser = subparsers.add_parser('batch', help='批量评估整个数据集')
    batch_parser.add_argument('--data_dir', type=str, required=True, help='视频文件夹路径')
    batch_parser.add_argument('--json', type=str, required=True, help='标注 JSON 文件路径')
    batch_parser.add_argument('--max_frames', type=int, default=16)
    batch_parser.add_argument('--output_dir', type=str, default='results')

    args = parser.parse_args()

    if args.mode == 'single':
        if not os.path.exists(args.video):
            print(f'[Error] Video not found: {args.video}')
            return 1
        try:
            diag = video_qa(args.video, args.question, max_frames_num=args.max_frames)
            answer = diag.get("final_answer", "")
            if args.output:
                os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(answer)
            print(f'\nFinal Answer: {answer}')
            return 0
        except Exception as e:
            print(f'[Error] {e}')
            import traceback; traceback.print_exc()
            return 1

    elif args.mode == 'batch':
        if not os.path.exists(args.data_dir):
            print(f'[Error] Data directory not found: {args.data_dir}')
            return 1
        if not os.path.exists(args.json):
            print(f'[Error] JSON file not found: {args.json}')
            return 1
        batch_evaluate(args.data_dir, args.json, output_dir=args.output_dir, max_frames_num=args.max_frames)
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit(main())
