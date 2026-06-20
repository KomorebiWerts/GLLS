import gradio as gr
import torch
import os
import sys
import asyncio
import numpy as np
import cv2
import socket
import json
import shutil
import tempfile
import base64
import zipfile
from datetime import datetime
from io import BytesIO
from collections import defaultdict
from PIL import Image

from kragad import paths as kragad_paths

from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    Qwen2_5_VLForConditionalGeneration, 
    LlavaOnevisionForConditionalGeneration
)

# ==========================================
# 0. Environment & Setup
# ==========================================
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
KRAGAD_PROJECT_ROOT = kragad_paths.project_root()
KRAGAD_DATA_ROOT = kragad_paths.data_root()
DEFAULT_MMAD_ROOT = kragad_paths.dataset_root()
DEFAULT_GRAPH_ROOT = kragad_paths.graph_cache_root()
DEFAULT_SAM3_PATH = kragad_paths.sam3_path()
DEFAULT_ABOUND_SAVE_PATH = kragad_paths.abound_save_path()
DEFAULT_VLM_MODEL_PATH = kragad_paths.vlm_model_path()

# Try importing vLLM
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
    print("✅ vLLM library detected.")
except ImportError:
    HAS_VLLM = False
    print("⚠️ vLLM not found. Acceleration disabled.")

# Import Project Modules
try:
    from kragad.models.mcts_sam import MCTSQuestionSample
    from kragad.models.localizer import ABounD_Localizer
    from kragad.rag.agent import SimInspecAgent
    from kragad.seg.sam3_engine import Sam3Engine
    # Import Loaders
    from kragad.data.dataset_loader import DSMVTecDatasetLoader, VisaDatasetLoader 
except ImportError as e:
    print(f"⚠️ Project module import failed: {e}")
    print("Ensure you are running this from the project root.")

# Qwen3 Support
try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False

# ==========================================
# 1. Utility Functions
# ==========================================
def find_free_port(start_port):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return ""

def _extract_path(item):
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("path") or item.get("name")
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return _extract_path(item[0])
    if hasattr(item, "name"):
        return getattr(item, "name")
    return None

def _pil_from_any(image_like):
    if image_like is None:
        return None
    if isinstance(image_like, Image.Image):
        return image_like
    if isinstance(image_like, np.ndarray):
        try:
            return Image.fromarray(image_like)
        except Exception:
            return None
    path = _extract_path(image_like)
    if path and os.path.exists(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None

def _zip_write_text(zf, arcname, text):
    data = _as_text(text).encode("utf-8", errors="replace")
    zf.writestr(arcname, data)

def _zip_write_pil_png(zf, arcname, pil_img):
    pil_img = _pil_from_any(pil_img)
    if pil_img is None:
        return False
    bio = BytesIO()
    pil_img.save(bio, format="PNG")
    zf.writestr(arcname, bio.getvalue())
    return True

def _zip_write_path(zf, src_path, arcname):
    if not src_path or not os.path.exists(src_path):
        return False
    try:
        zf.write(src_path, arcname=arcname)
        return True
    except Exception:
        return False

def _normalize_gallery_items(gallery_value):
    if not gallery_value:
        return []
    if not isinstance(gallery_value, list):
        return []
    normalized = []
    for item in gallery_value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append({"image": item[0], "label": _as_text(item[1])})
        else:
            normalized.append({"image": item, "label": ""})
    return normalized

def _maybe_generate_docx_report(
    *,
    out_docx_path,
    timestamp_iso,
    category,
    task_type,
    question,
    ground_truth,
    options,
    rag_text,
    final_diagnosis_md,
    phase2_prompt,
    phase1_prompt_md,
    logs,
    input_image_pil,
    heatmap_pil,
    logic_view_pil,
    local_refinement_paths,
    rag_image_paths,
):
    try:
        from docx import Document  # type: ignore
        from docx.shared import Inches  # type: ignore
    except Exception as e:
        return False, f"python-docx not available: {e}"

    def _add_picture(doc, pil_img=None, path=None, caption=None, width_inches=6.0):
        if caption:
            doc.add_paragraph(_as_text(caption))
        if path and os.path.exists(path):
            doc.add_picture(path, width=Inches(width_inches))
            return True
        pil_img = _pil_from_any(pil_img)
        if pil_img is None:
            return False
        tmp_png = os.path.join(os.path.dirname(out_docx_path), f"_tmp_{os.urandom(6).hex()}.png")
        pil_img.save(tmp_png)
        doc.add_picture(tmp_png, width=Inches(width_inches))
        return True

    try:
        doc = Document()
        doc.add_heading("Sim-Inspec Run Report", level=0)
        doc.add_paragraph(f"Timestamp: {timestamp_iso}")
        if category:
            doc.add_paragraph(f"Category: {category}")
        if task_type:
            doc.add_paragraph(f"Task Type: {task_type}")

        doc.add_heading("Question", level=1)
        doc.add_paragraph(_as_text(question))
        if ground_truth:
            doc.add_paragraph(f"Ground Truth: {ground_truth}")
        if options:
            doc.add_paragraph("Options (JSON):")
            doc.add_paragraph(json.dumps(options, ensure_ascii=False, indent=2))

        doc.add_heading("Input Image", level=1)
        _add_picture(doc, pil_img=input_image_pil, caption="Input View")

        doc.add_heading("RAG Knowledge", level=1)
        doc.add_heading("Text Rules", level=2)
        doc.add_paragraph(_as_text(rag_text))
        if rag_image_paths:
            doc.add_heading("Visual Standards", level=2)
            for i, p in enumerate(rag_image_paths, start=1):
                _add_picture(doc, path=p, caption=f"RAG Image {i}: {os.path.basename(p)}", width_inches=5.5)

        doc.add_heading("Global Anomaly Heatmap", level=1)
        _add_picture(doc, pil_img=heatmap_pil, caption="Global Anomaly Heatmap (Overlay)")

        doc.add_heading("Local Refinement", level=1)
        for i, p in enumerate(local_refinement_paths or [], start=1):
            _add_picture(doc, path=p, caption=f"Refinement View {i}: {os.path.basename(p)}", width_inches=5.5)

        doc.add_heading("Logic Debug", level=1)
        _add_picture(doc, pil_img=logic_view_pil, caption="Logic Debug View")
        doc.add_heading("Phase 1 Prompt", level=2)
        doc.add_paragraph(_as_text(phase1_prompt_md))

        doc.add_heading("Final", level=1)
        doc.add_heading("Final Prompt (Phase 2)", level=2)
        doc.add_paragraph(_as_text(phase2_prompt))
        doc.add_heading("Final Diagnosis", level=2)
        doc.add_paragraph(_as_text(final_diagnosis_md))

        doc.add_heading("Logs", level=1)
        doc.add_paragraph(_as_text(logs))

        doc.save(out_docx_path)
        return True, "ok"
    except Exception as e:
        return False, f"docx generation failed: {e}"

def export_current_run_bundle(
    input_image,
    question,
    ground_truth,
    category,
    task_type,
    options,
    rag_gallery,
    rag_text,
    heatmap_image,
    sam3_gallery,
    focus_files,
    final_diagnosis_md,
    phase2_prompt,
    logic_view_image,
    phase1_prompt_md,
    logs_text,
    include_docx,
):
    final_diagnosis_md = _as_text(final_diagnosis_md).strip()
    if not final_diagnosis_md or "Waiting for results" in final_diagnosis_md:
        return None, "⚠️ No completed run to export yet."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_iso = datetime.now().isoformat(timespec="seconds")
    export_root = tempfile.mkdtemp(prefix="siminspec_export_")
    zip_path = os.path.join(export_root, f"siminspec_run_{timestamp}.zip")
    top_dir = f"siminspec_run_{timestamp}"

    rag_image_paths = []
    if rag_gallery:
        for item in rag_gallery:
            p = _extract_path(item)
            if p and os.path.exists(p):
                rag_image_paths.append(p)

    local_refinement_paths = []
    if focus_files:
        for item in focus_files:
            p = _extract_path(item)
            if p and os.path.exists(p):
                local_refinement_paths.append(p)

    sam3_items = _normalize_gallery_items(sam3_gallery)

    manifest = {
        "export_version": 1,
        "timestamp": timestamp_iso,
        "category": _as_text(category),
        "task_type": _as_text(task_type),
        "question": _as_text(question),
        "ground_truth": _as_text(ground_truth),
        "options": options if isinstance(options, dict) else {},
        "has_docx": False,
    }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _zip_write_text(
            zf,
            f"{top_dir}/README.txt",
            "This ZIP contains one Sim-Inspec agent run export.\n"
            "Folders:\n"
            "  question/  rag/  global/  local_refinement/  logic_debug/  final/  logs/\n",
        )

        # Question / metadata
        _zip_write_text(zf, f"{top_dir}/question/question.txt", question)
        _zip_write_text(zf, f"{top_dir}/question/ground_truth.txt", ground_truth)
        if isinstance(options, dict) and options:
            _zip_write_text(zf, f"{top_dir}/question/options.json", json.dumps(options, ensure_ascii=False, indent=2))
        if _pil_from_any(input_image) is not None:
            _zip_write_pil_png(zf, f"{top_dir}/question/input_view.png", input_image)

        # RAG
        _zip_write_text(zf, f"{top_dir}/rag/text_rules.txt", rag_text)
        for i, p in enumerate(rag_image_paths, start=1):
            base = os.path.basename(p) or f"rag_{i}.png"
            _zip_write_path(zf, p, f"{top_dir}/rag/visual_standards/{i:02d}_{base}")

        # Global
        _zip_write_pil_png(zf, f"{top_dir}/global/global_anomaly_heatmap.png", heatmap_image)

        # Local refinement (files)
        for i, p in enumerate(local_refinement_paths, start=1):
            base = os.path.basename(p) or f"refinement_{i}.png"
            _zip_write_path(zf, p, f"{top_dir}/local_refinement/{i:02d}_{base}")

        # Local refinement (gallery snapshots)
        if sam3_items:
            sam3_index = []
            for i, it in enumerate(sam3_items, start=1):
                label = _as_text(it.get("label", ""))
                arc = f"{top_dir}/local_refinement/gallery_{i:02d}.png"
                ok = _zip_write_pil_png(zf, arc, it.get("image"))
                if ok:
                    sam3_index.append({"file": f"local_refinement/gallery_{i:02d}.png", "label": label})
            if sam3_index:
                _zip_write_text(zf, f"{top_dir}/local_refinement/gallery_index.json", json.dumps(sam3_index, ensure_ascii=False, indent=2))

        # Logic debug
        _zip_write_pil_png(zf, f"{top_dir}/logic_debug/view_image.png", logic_view_image)
        _zip_write_text(zf, f"{top_dir}/logic_debug/phase1_prompt.md", phase1_prompt_md)

        # Final
        _zip_write_text(zf, f"{top_dir}/final/final_prompt.txt", phase2_prompt)
        _zip_write_text(zf, f"{top_dir}/final/final_diagnosis.md", final_diagnosis_md)

        # Logs
        _zip_write_text(zf, f"{top_dir}/logs/live_logs.txt", logs_text)

        # Optional DOCX
        include_docx = bool(include_docx)
        if include_docx:
            docx_path = os.path.join(export_root, f"siminspec_run_{timestamp}.docx")
            ok, msg = _maybe_generate_docx_report(
                out_docx_path=docx_path,
                timestamp_iso=timestamp_iso,
                category=category,
                task_type=task_type,
                question=question,
                ground_truth=ground_truth,
                options=options if isinstance(options, dict) else {},
                rag_text=rag_text,
                final_diagnosis_md=final_diagnosis_md,
                phase2_prompt=phase2_prompt,
                phase1_prompt_md=phase1_prompt_md,
                logs=logs_text,
                input_image_pil=input_image,
                heatmap_pil=heatmap_image,
                logic_view_pil=logic_view_image,
                local_refinement_paths=local_refinement_paths,
                rag_image_paths=rag_image_paths,
            )
            if ok and os.path.exists(docx_path):
                manifest["has_docx"] = True
                _zip_write_path(zf, docx_path, f"{top_dir}/report.docx")
            else:
                _zip_write_text(zf, f"{top_dir}/report_docx_error.txt", msg)

        # Manifest (write once, after optional DOCX)
        _zip_write_text(zf, f"{top_dir}/manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    docx_note = " (with report.docx)" if manifest.get("has_docx") else ""
    return zip_path, f"✅ Export ready: {os.path.basename(zip_path)}{docx_note}"

# ==========================================
# 2. Dataset Manager (INTELLIGENT PATH FIX)
# ==========================================
class DatasetManager:
    def __init__(self, root_path, dataset_name="mvtec"):
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.loader = None
        self.all_samples = []          
        self.subfolder_choices = []    
        self.task_type_choices = []    
        
    def load_subclass(self, category):
        """Loads category data with auto-path correction."""
        
        # --- 1. Intelligent Path Resolution ---
        # Normalize path to remove trailing slashes
        raw_root = os.path.normpath(self.root_path)
        
        dataset_folder_map = {
            'visa': 'VisA',
            'mvtec': 'DS-MVTec'
        }
        target_folder = dataset_folder_map.get(self.dataset_name.lower())

        if not target_folder:
            return f"❌ Unknown dataset type: {self.dataset_name}"

        # Check if root already ends with specific folder, if not, append it
        if os.path.basename(raw_root) == target_folder:
            final_root = raw_root
        else:
            final_root = os.path.join(raw_root, target_folder)

        # Validation
        if not os.path.exists(final_root):
            return f"❌ Dataset path not found: {final_root}. Please check 'Dataset Root' config."

        # --- 2. Loader Initialization ---
        # Only re-initialize if loader is missing or path changed
        if not self.loader: 
            if self.dataset_name.lower() == "visa":
                print(f"📂 Initializing VisaDatasetLoader at {final_root}")
                self.loader = VisaDatasetLoader(root_path=final_root)
            else:
                print(f"📂 Initializing DSMVTecDatasetLoader at {final_root}")
                self.loader = DSMVTecDatasetLoader(root_path=final_root)
            
        try:
            # The loader now looks inside ".../VisA/candle" correctly
            self.all_samples = list(self.loader.parse_samples(category))
            
            subfolders = set()
            task_types = set()
            for s in self.all_samples:
                path_parts = s.get('image_path', '').replace('\\', '/').split('/')
                # Simple subfolder parsing (usually 2nd to last)
                if len(path_parts) >= 2:
                    subfolders.add(path_parts[-2])
                task_types.add(s.get('task_type', 'Unknown'))
            
            self.subfolder_choices = sorted(list(subfolders))
            self.task_type_choices = sorted(list(task_types))
            
            return f"✅ [{self.dataset_name.upper()}] '{category}' loaded: {len(self.all_samples)} samples from {target_folder}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Load Failed: {str(e)}"

    def filter_samples(self, subfolder=None, task_type=None, search_query=""):
        choices = []
        for idx, s in enumerate(self.all_samples):
            path_match = True
            if subfolder and subfolder != "All":
                path_match = subfolder in s.get('image_path', '')
            
            type_match = True
            if task_type and task_type != "All":
                type_match = (s.get('task_type') == task_type)
            
            img_path = s.get('image_path', '')
            file_name = os.path.basename(img_path) if img_path else "NoImg"

            search_match = True
            if search_query:
                q_text = s.get('question', '').lower()
                search_match = (search_query.lower() in q_text) or (search_query.lower() in file_name.lower())

            if path_match and type_match and search_match:
                short_q = s['question'][:40].replace("\n", " ") 
                gt = s.get('gt_answer', '?')
                choices.append(f"{idx} | [{file_name}] {short_q}... (GT: {gt})")
        
        return choices

    def get_sample_by_idx(self, global_idx):
        if 0 <= global_idx < len(self.all_samples):
            s = self.all_samples[global_idx]
            img_path = s['image_path']
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                # Ensure options is a dict
                opts = s.get('options', {})
                if isinstance(opts, str):
                    try:
                        opts = json.loads(opts)
                    except:
                        opts = {}
                return img, s['question'], s['task_type'], s.get('gt_answer', 'Unknown'), opts
        return None, "", "", "", {}

# ==========================================
# 3. Unified VLM Inference
# ==========================================
class UnifiedVLMInference:
    def __init__(self, model_path, device, model_type="qwen2.5-vl", use_vllm=False, gpu_util=0.85):
        self.device = device 
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.use_vllm = use_vllm and HAS_VLLM
        
        print(f"🚀 Loading VLM: {self.model_type} | vLLM Enabled: {self.use_vllm}")

        # --- vLLM Backend ---
        if self.use_vllm:
            print(f"⚡ Initializing vLLM Engine (GPU Util: {gpu_util})...")
            try:
                self.model = LLM(
                    model=model_path,
                    trust_remote_code=True,
                    gpu_memory_utilization=gpu_util, 
                    tensor_parallel_size=1,
                    dtype="bfloat16",       
                    max_model_len=8192,     
                    enforce_eager=False,
                    disable_custom_all_reduce=True
                )
                self.sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
                print("✅ vLLM Engine Ready!")
                return
            except Exception as e:
                print(f"❌ vLLM Init Failed: {e}")
                print("Fallback to Transformer mode...")
                self.use_vllm = False

        # --- Transformers Backend Fallback ---
        if "qwen3" in self.model_type:
            if not HAS_QWEN3:
                raise ImportError("Qwen3 class not available.")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", device_map=device, trust_remote_code=True
            )
        elif "qwen" in self.model_type: # qwen2.5-vl
            self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
            )
        elif "llava" in self.model_type:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=device
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if hasattr(self, 'model'):
            self.model.eval()

    def generate(self, content_list, max_tokens=4096):
        clean_content = []
        pil_images = []
        
        for item in content_list:
            if item['type'] == 'text':
                clean_content.append(item)
            elif item['type'] == 'image':
                img = item['image']
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img).convert("RGB")
                elif isinstance(img, str):
                    if os.path.exists(img):
                        img = Image.open(img).convert("RGB")
                    else:
                        continue
                pil_images.append(img)
                clean_content.append({"type": "image", "image": img})

        if self.use_vllm:
            return self._generate_vllm(clean_content, max_tokens)
        elif "qwen" in self.model_type:
            return self._generate_qwen_transformers(clean_content, max_tokens)
        elif "llava" in self.model_type:
            return self._generate_llava_transformers(clean_content, pil_images, max_tokens)
        
        return "Model backend not supported."

    def _generate_vllm(self, content, max_tokens):
        vllm_messages = []   
        for item in content:
            if item["type"] == "text":
                vllm_messages.append(item)
            elif item["type"] == "image":
                base64_str = image_to_base64(item["image"])
                vllm_messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_str}"
                    }
                })

        if max_tokens != self.sampling_params.max_tokens:
            self.sampling_params.max_tokens = max_tokens
        
        messages = [{"role": "user", "content": vllm_messages}]
        outputs = self.model.chat(
            messages=messages,
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        return outputs[0].outputs[0].text

    def _generate_qwen_transformers(self, content, max_tokens):
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = []
        for item in content:
            if item["type"] == "image":
                image_inputs.append(item["image"])
        
        if not image_inputs:
            image_inputs = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=0,
                do_sample=False 
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def _generate_llava_transformers(self, content, pil_images, max_tokens):
        llava_content = []
        for item in content:
            if item['type'] == 'text':
                llava_content.append(item)
            elif item['type'] == 'image':
                llava_content.append({"type": "image"}) 
        
        conversation = [{"role": "user", "content": llava_content}]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        inputs = self.processor(
            text=text_prompt, 
            images=pil_images if pil_images else None, 
            return_tensors='pt'
        ).to(self.device, torch.float16)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=0,
                do_sample=False 
            )
        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        if "assistant\n" in decoded:
            return decoded.split("assistant\n")[-1]
        return decoded

# ==========================================
# 4. Global System Controller
# ==========================================
class GlobalSystem:
    def __init__(self):
        self.vlm = None
        self.localizer = None
        self.rag_agent = None
        self.sam_engine = None
        self.args = None
        self.data_manager = None
        self.is_initialized = False
        self.current_dataset = None
        self.current_ckpt = None

    def initialize(self, model_path, ckpt_path, graph_root, sam_path, dataset_root, gpu_id, model_type, use_vllm, gpu_util, dataset_name):
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = f"cuda:{gpu_id}"
            device_id_int = int(gpu_id)
            
            print(f"🔧 System Config: Target GPU {gpu_id} | Logical Device: {device} | Dataset: {dataset_name}")

            # Initialize DatasetManager with the raw path; it handles resolution internally
            self.data_manager = DatasetManager(dataset_root, dataset_name)
            
            # Re-init VLM if params changed
            if self.vlm is None or self.vlm.model_type != model_type or self.vlm.model_path != model_path or self.vlm.use_vllm != use_vllm:
                if self.vlm: 
                    print("🔄 Reloading VLM... freeing memory.")
                    del self.vlm
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                self.vlm = UnifiedVLMInference(
                    model_path, 
                    device, 
                    model_type, 
                    use_vllm=use_vllm, 
                    gpu_util=gpu_util
                )
            
            # Check if Localizer needs reload
            localizer_needs_reload = (
                self.localizer is None or 
                self.current_dataset != dataset_name or 
                self.current_ckpt != ckpt_path
            )

            if localizer_needs_reload:
                print(f"🔄 (Re)Loading Localizer for dataset: {dataset_name}...")
                if self.localizer:
                    del self.localizer
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                class Args: pass
                self.args = Args()
                self.args.dataset = dataset_name 
                self.args.image_size = 336
                self.args.checkpoint_path = ckpt_path 
                self.args.save_path = ckpt_path 
                self.args.k_shot = 4
                
                # Default params
                self.args.features_list = [6, 12, 18, 24]
                self.args.num_visual_finetune_layers = 12
                self.args.depth = 7; self.args.n_ctx = 11; self.args.spe = 4
                self.args.w0 = 0.15; self.args.w1 = 0.35; self.args.w2 = 0.35; self.args.w3 = 0.15
                
                self.localizer = ABounD_Localizer(self.args, device_id=device_id_int)
                
                self.current_dataset = dataset_name
                self.current_ckpt = ckpt_path
            
            if not self.rag_agent: 
                self.rag_agent = SimInspecAgent(None, graph_root, k_shot=1)
            
            if not self.sam_engine and os.path.exists(sam_path): 
                print(f"🔌 Loading SAM3 to {device}...")
                self.sam_engine = Sam3Engine(sam_path, device=device)
            
            self.is_initialized = True
            vllm_status = f"ON (Util: {gpu_util})" if (use_vllm and self.vlm.use_vllm) else "OFF"
            return f"✅ System Ready (GPU:{gpu_id}) | Model: {model_type} | Data: {dataset_name}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Init Failed: {str(e)}"

global_sys = GlobalSystem()

class StreamableMCTS(MCTSQuestionSample):
    def __init__(self, row, args, inference_engine, localizer, rag_agent, rag_blocks, sam_engine, update_callback):
        super().__init__(row, args, inference_engine, localizer, rag_agent=rag_agent, rag_blocks=rag_blocks, sam_engine=sam_engine)
        self.update_callback = update_callback 

    async def get_anomaly_heatmap(self, image):
        heatmap, obj_name = await super().get_anomaly_heatmap(image)
        
        if heatmap is None or np.sum(heatmap) == 0:
             await self.update_callback("log", "⚠️ Warning: Localizer returned empty heatmap (check dataset config).")
             return heatmap, obj_name

        try:
            hm_min = heatmap.min()
            hm_max = heatmap.max()
            denominator = hm_max - hm_min + 1e-8
            
            hm_norm = (heatmap - hm_min) / denominator
            hm_uint8 = np.uint8(255 * hm_norm)
            hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
            hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
            
            img_np = np.array(image.resize((self.image_width, self.image_height)))
            heatmap_resized = cv2.resize(hm_color, (self.image_width, self.image_height))
            
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_resized, 0.4, 0)
            
            await self.update_callback("heatmap", Image.fromarray(overlay))
        except Exception as e:
            print(f"Heatmap Vis Error: {e}")
            
        return heatmap, obj_name

    async def simulation(self, node):
        score = await super().simulation(node)
        if node.state['depth'] > 0:
            await self.update_callback("log", f"🔍 MCTS D{node.state['depth']} Searching... Score: {score:.2f}")
        return score

# ==========================================
# 5. Async Logic Stream
# ==========================================

async def run_analysis_stream(image, question, subclass, task_type, options, override_rag_text=None, override_rag_files=None):
    if not global_sys.is_initialized:
        yield None, "", None, None, None, "⚠️ System Not Initialized", "", "", None, None, None, ""
        return
    
    # Debug Options Passing
    if not options:
        print("⚠️ [DEBUG] No options received in run_analysis_stream!")
    else:
        print(f"✅ [DEBUG] Options received: {len(options)} keys")

    # --- RAG Loading Logic (Updated for Smart Splitting) ---
    rag_blocks = []
    
    if override_rag_text or (override_rag_files is not None and len(override_rag_files) > 0):
        print("🔧 Using Manual RAG Context Overrides (Auto-Splitting Global/Local)")
        
        # 1. Bucket sorting for images based on filename
        global_imgs = []
        local_imgs = []
        
        if override_rag_files:
            for item in override_rag_files:
                path = None
                # Extract path safely
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    path = item[0]
                elif isinstance(item, dict):
                    path = item.get('name') or item.get('path')
                elif hasattr(item, 'name'):
                    path = item.name
                elif isinstance(item, str):
                    path = item
                
                if path and os.path.exists(path):
                    filename = os.path.basename(path).lower()
                    # Backend Logic: Checks if 'whole' or 'global' is in region name
                    if "whole" in filename or "global" in filename:
                        global_imgs.append(path)
                    else:
                        local_imgs.append(path)

        # 2. Construct Blocks
        # Block A: Global/Whole (Targeted by Phase 1)
        # We attach text here so Phase 1 knows what "Normal" looks like
        rag_blocks.append({
            "region": "Manual Global Reference (Whole Object)", 
            "text": override_rag_text if override_rag_text else "",
            "images": global_imgs
        })

        # Block B: Local/Regions (Targeted by Phase 2)
        # We DUPLICATE text here so Phase 2 knows "Defect" definitions
        # (Phase 2 explicitly skips blocks named 'whole', so it needs its own copy)
        rag_blocks.append({
            "region": "Manual Local Defect Reference",
            "text": override_rag_text if override_rag_text else "",
            "images": local_imgs
        })
    else:
        rag_blocks = global_sys.rag_agent.get_rag_context(subclass)

    # Prepare Display Data
    rag_file_paths, rag_text_content = [], ""
    if rag_blocks:
        for i, b in enumerate(rag_blocks):
            t_content = b.get('text', '').strip()
            if t_content:
                rag_text_content += f"=== {b.get('region', f'Region {i+1}')} ===\n{t_content}\n\n"
            
            if b.get('images'):
                for img_path in b['images']:
                    if os.path.exists(img_path): rag_file_paths.append(img_path)
    
    state = {"heatmap": None, "log": "🚀 Starting MCTS Visual Trace..."}
    if override_rag_text:
        state["log"] += "\n⚠️ NOTE: Running with MODIFIED Knowledge Base..."
    if global_sys.vlm.use_vllm:
        state["log"] += "\n⚡ vLLM Acceleration: ACTIVE"

    # Yield 1: Init
    yield rag_file_paths, rag_text_content, None, None, None, state["log"], "", "", None, None, None, ""

    async def callback(key, value):
        if key == "heatmap": state["heatmap"] = value
        if key == "log": state["log"] += f"\n{value}"

    row = {'image': image.convert("RGB"), 'question': question, 'options': options, 'category': subclass, 'type': task_type}
    try:
        mcts = StreamableMCTS(row, global_sys.args, global_sys.vlm, global_sys.localizer, global_sys.rag_agent, rag_blocks, global_sys.sam_engine, callback)
        mcts_task = asyncio.create_task(mcts.process())
        
        while not mcts_task.done():
            await asyncio.sleep(0.1)
            yield rag_file_paths, rag_text_content, state["heatmap"], None, None, state["log"], "", "", None, None, None, ""
        
        result = await mcts_task
        
        # Save crops for gallery
        temp_dir = tempfile.mkdtemp()
        crop_file_paths = []
        gallery_items = []
        
        if result.get("red_box_image"):
            rb_path = os.path.join(temp_dir, "global_trace.png")
            result["red_box_image"].save(rb_path)
            crop_file_paths.append(rb_path)
            gallery_items.append((result["red_box_image"], "Global Scope"))
            
        for i, crop in enumerate(result.get("crop_images", [])):
            c_path = os.path.join(temp_dir, f"focus_crop_{i}.png")
            crop.save(c_path)
            crop_file_paths.append(c_path)
            gallery_items.append((crop, f"Focus View {i+1}"))
        
        debug_info = result.get("logic_debug_info", {})
        debug_view_img = debug_info.get("view_image")
        raw_p1_prompt = debug_info.get("phase1_prompt", "")
        
        formatted_prompt = raw_p1_prompt
        if formatted_prompt:
            formatted_prompt = formatted_prompt.replace("=== PHASE 1: GLOBAL INSPECTION ===", "### 🚩 PHASE 1: GLOBAL INSPECTION")
            formatted_prompt = formatted_prompt.replace("STEP 1:", "\n**STEP 1:**").replace("STEP 2:", "\n**STEP 2:**").replace("STEP 3:", "\n**STEP 3:**")
            formatted_prompt = formatted_prompt.replace("INSTRUCTION:", "\n#### 📝 INSTRUCTION:")
            formatted_prompt = formatted_prompt.replace("[SPECIAL INSPECTION VIEW]", "**[👁️ SPECIAL INSPECTION VIEW]**")
        else:
            formatted_prompt = "*No prompt data available yet.*"
        
        # Build Final Prompt
        final_content = []
        if result.get("rag_content_list"): final_content.extend(result["rag_content_list"])
        if result.get("red_box_image"):
            final_content.append({"type": "text", "text": "Image: Global Trace (Red Box)\n"}); final_content.append({"type": "image", "image": result["red_box_image"]})
        for i, crop in enumerate(result.get("crop_images", [])):
            final_content.append({"type": "text", "text": f"Image: Local Refinement {i+1}\n"}); final_content.append({"type": "image", "image": crop})
        
        final_prompt = result.get("prompt")
        final_content.append({"type": "text", "text": final_prompt})
        
        state["log"] += "\n🧠 Generating Final Diagnosis..."
        final_response = global_sys.vlm.generate(final_content)
        state["log"] += "\n✅ Analysis Complete."
        
        yield (
            rag_file_paths,     
            rag_text_content,   
            state["heatmap"],   
            gallery_items,      
            crop_file_paths,    
            state["log"],       
            final_response,     
            final_prompt,       
            result.get("red_box_image"), 
            crop_file_paths,
            debug_view_img,
            formatted_prompt
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield rag_file_paths, rag_text_content, state["heatmap"], None, None, f"❌ Error: {str(e)}", "", "", None, None, None, ""

# Wrapper for Gradio Generator
def search_runner_wrapper(img, q, sub, task, opts, man_txt=None, man_files=None):
    # Verify opts before running
    if opts is None or not isinstance(opts, dict):
        print("⚠️ [WRAPPER] Options are None or Invalid, resetting to empty dict.")
        opts = {}
        
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    gen = run_analysis_stream(img, q, sub, task, opts, man_txt, man_files)
    try:
        while True: yield loop.run_until_complete(gen.__anext__())
    except StopAsyncIteration: pass
    finally: loop.close()

# Phase 2 Only Rerun
def run_manual_inference_phase2(rag_file_objs, rag_text, prompt, state_red_box, focus_file_objs):
    if not global_sys.vlm: return "⚠️ System Not Initialized", ""
    content_list = [{"type": "text", "text": "## MANUAL REFINEMENT MODE (PHASE 2) ##\n"}]
    
    if rag_file_objs:
        content_list.append({"type": "text", "text": "=== VISUAL CRITERIA ===\n"})
        for item in rag_file_objs:
            path = None
            if isinstance(item, (list, tuple)) and len(item) > 0: path = item[0]
            elif isinstance(item, dict): path = item.get('name') or item.get('path')
            elif hasattr(item, 'name'): path = item.name
            elif isinstance(item, str): path = item
            
            if path and os.path.exists(path):
                content_list.append({"type": "image", "image": Image.open(path).convert("RGB")})
    
    content_list.append({"type": "text", "text": f"\n=== TEXT RULES ===\n{rag_text}\n"})
    
    if focus_file_objs:
        content_list.append({"type": "text", "text": "\n=== TARGET VIEWS (MANUAL/SAM3) ===\n"})
        for i, item in enumerate(focus_file_objs):
            path = None
            if isinstance(item, (list, tuple)) and len(item) > 0: path = item[0]
            elif isinstance(item, dict): path = item.get('name') or item.get('path')
            elif hasattr(item, 'name'): path = item.name
            elif isinstance(item, str): path = item

            if path and os.path.exists(path):
                content_list.append({"type": "text", "text": f"Inspection View {i+1}:\n"})
                content_list.append({"type": "image", "image": Image.open(path).convert("RGB")})
    elif state_red_box:
        content_list.append({"type": "image", "image": state_red_box})
        
    content_list.append({"type": "text", "text": f"\n--- INSTRUCTION ---\n{prompt}"})
    
    try:
        res = global_sys.vlm.generate(content_list)
        return res, "✅ Phase 2 Rerun Success"
    except Exception as e: return f"❌ Failed: {e}", f"Error: {e}"

# ==========================================
# 6. Gradio UI
# ==========================================

CSS = """
:root {
    --nju-purple: #5D0F52;
    --nju-purple-dark: #3d0a35;
    --bg-light: #f8f6f8;
}
body { background-color: var(--bg-light); font-family: 'Segoe UI', sans-serif; }
.header-bar {
    background: linear-gradient(135deg, var(--nju-purple-dark) 0%, var(--nju-purple) 100%);
    padding: 2.2rem 4rem;
    border-radius: 0 0 35px 35px;
    box-shadow: 0 10px 30px rgba(93, 15, 82, 0.25);
    margin-bottom: 2rem;
}
.header-bar * { color: white !important; }
.header-bar h1 { font-weight: 900; font-size: 2.6rem; margin: 0; }
.header-bar p { font-size: 1.2rem; opacity: 0.9; margin-top: 0.5rem; }
.nju-image-upload .upload-button, .nju-image-upload .flex.flex-col, .nju-image-upload svg { 
    color: var(--nju-purple) !important; fill: var(--nju-purple) !important; 
}
.nju-image-upload .gr-box { border-color: var(--nju-purple) !important; }
.custom-card {
    background: white; border-radius: 24px; padding: 2rem;
    border: 1.5px solid rgba(93, 15, 82, 0.08); box-shadow: 0 5px 20px rgba(0,0,0,0.02);
    transition: transform 0.3s ease;
}
.custom-card:hover { transform: translateY(-5px); box-shadow: 0 15px 40px rgba(93, 15, 82, 0.08); }
.section-title {
    color: var(--nju-purple); font-weight: 800; font-size: 1.4rem;
    border-left: 8px solid var(--nju-purple); padding-left: 18px; margin-bottom: 1.8rem;
}
.section-title-sm {
    color: var(--nju-purple); font-weight: 700; font-size: 1.1rem; margin-bottom: 0.8rem;
}
.nju-btn-primary {
    background: var(--nju-purple) !important; color: white !important; font-weight: 700 !important;
    border-radius: 14px !important; box-shadow: 0 6px 15px rgba(93, 15, 82, 0.2) !important;
}
.nju-btn-primary:hover { filter: brightness(1.2) !important; transform: scale(1.03); }
.log-box textarea {
    background: #1d1d1f !important; color: #32d74b !important; font-family: 'Fira Code', monospace !important; border-radius: 18px !important;
}
.prompt-card {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px;
    height: 400px !important; overflow-y: auto; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    font-family: 'Segoe UI', system-ui, sans-serif; font-size: 0.95rem; line-height: 1.6;
}
.prompt-card h3 { color: var(--nju-purple); border-bottom: 2px solid #f0f0f0; padding-bottom: 8px; margin-top: 10px; }
.prompt-card strong { color: #2d3748; background-color: #f3e8f2; padding: 0 4px; border-radius: 4px; }
.logic-view-img {
    height: 400px !important; border-radius: 12px; overflow: hidden; border: 1px solid #e5e7eb;
    display: flex; justify-content: center; align-items: center; background: #f9f9f9;
}
"""

def create_ui():
    nju_theme = gr.themes.Soft(primary_hue="purple", neutral_hue="slate").set(block_radius="24px", button_primary_background_fill="#5D0F52")

    MVTEC_CLASSES = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", 
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    VISA_CLASSES = [
        "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", 
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    ]
    
    with gr.Blocks(css=CSS, theme=nju_theme, title="NJU Sim-Inspec Agent") as demo:
        
        # --- Header ---
        with gr.Row(elem_classes="header-bar"):
            with gr.Column(scale=4):
                gr.Markdown("# 🏛️ Sim-Inspec: NJU Intelligent Inspection")
                gr.Markdown("Agentic Industrial Anomaly Detection Framework | 南京大学智能科学与技术学院")
            with gr.Column(scale=1):
                status_box = gr.Textbox(label="System Status", value="Uninitialized", interactive=False, container=False)

        # --- Main Layout ---
        with gr.Row():
            
            # --- LEFT: Config & Search ---
            with gr.Column(scale=1, min_width=440):
                with gr.Column(elem_classes="custom-card"):
                    gr.Markdown("### ⚙️ Engine Configuration", elem_classes="section-title")
                    with gr.Accordion("Model & GPU Settings", open=True):
                        dd_model_type = gr.Dropdown(
                            ["qwen2.5-vl", "qwen3-vl", "llava-onevision"], 
                            label="Model Architecture", value="qwen3-vl"
                        )
                        with gr.Row():
                            cb_vllm = gr.Checkbox(label="Enable vLLM Acceleration", value=True, interactive=True)
                            dd_gpu = gr.Dropdown([str(i) for i in range(8)], value="4", label="GPU ID")
                        
                        sl_gpu_util = gr.Slider(0.3, 0.95, value=0.85, step=0.05, label="vLLM GPU Memory Utilization")

                        p_model = gr.Textbox(DEFAULT_VLM_MODEL_PATH, label="VLM Path")
                        p_sam = gr.Textbox(DEFAULT_SAM3_PATH, label="SAM3 Path")

                        # [Dataset Selection]
                        dd_dataset = gr.Dropdown(["mvtec", "visa"], label="Dataset Name", value="visa")

                        # [Path Auto-Correction Demo]
                        # Default set to parent folder MMAD to show auto-append feature
                        p_data = gr.Textbox(DEFAULT_MMAD_ROOT, label="Dataset Root (Parent Folder)")

                        p_ckpt = gr.Textbox(
                            DEFAULT_ABOUND_SAVE_PATH,
                            label="Checkpoint Path (Folder)"
                        )

                        p_graph = gr.Textbox(DEFAULT_GRAPH_ROOT, label="Graph DB")
                    
                    btn_init = gr.Button("🚀 Initialize Engine", elem_classes="nju-btn-primary")
                
                with gr.Column(elem_classes="custom-card"):
                    gr.Markdown("### 📂 Case Manager", elem_classes="section-title")
                    
                    dd_cat = gr.Dropdown(VISA_CLASSES, label="1. Category", value="candle", interactive=True)
                    
                    btn_load = gr.Button("📥 Load Dataset", size="sm")
                    
                    with gr.Row():
                        dd_sub_type = gr.Dropdown(label="2. Subclass", choices=["All"], value="All")
                        dd_logic = gr.Dropdown(label="3. Task Type", choices=["All"], value="All")
                    
                    txt_search_case = gr.Textbox(placeholder="🔍 Search question...", label="4. Global Search")
                    dd_samples_list = gr.Dropdown(label="5. Select Sample", choices=[], interactive=True)
                    
                    img_preview_in = gr.Image(label="Input View", type="pil", height=280, elem_classes="nju-image-upload")
                    txt_q_in = gr.Textbox(label="Inspector Command", lines=2)
                    with gr.Accordion("Metadata", open=False):
                        txt_gt_out = gr.Textbox(label="Ground Truth", interactive=False)
                        txt_opts_json = gr.Code(label="Options", language="json")

                    btn_run_main = gr.Button("✨ Start Agent Inspection (Auto)", variant="primary", size="lg", elem_classes="nju-btn-primary")

            # --- RIGHT: Visual Intelligence Dashboard ---
            with gr.Column(scale=2):
                
                # Top Intelligence Row
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="custom-card"):
                        gr.Markdown("### 📚 Knowledge Standard (RAG)", elem_classes="section-title")
                        gr.Markdown("*Edit below then click 'Rerun Full Pipeline' to test new rules*", elem_classes="text-sm text-gray-500")
                        with gr.Tabs():
                            with gr.TabItem("🖼️ Visual Standards"):
                                gal_rag_source = gr.Gallery(
                                    label="RAG Retrieved Images", 
                                    show_label=True,
                                    columns=3, 
                                    rows=2,
                                    height=250, 
                                    object_fit="contain", 
                                    interactive=True,
                                    type="filepath"
                                )
                            with gr.TabItem("📝 Text Rules"):
                                txt_rag_manual = gr.TextArea(label="Instruction Text", lines=10, interactive=True)
                        
                        btn_rerun_all = gr.Button("🔄 Rerun Full Pipeline (With Edited Knowledge)", size="sm", variant="secondary")
                    
                    with gr.Column(scale=1, elem_classes="custom-card"):
                        gr.Markdown("### 🔥 Global Anomaly Heatmap", elem_classes="section-title")
                        img_hm_out = gr.Image(label="Anomaly Trace", type="pil", height=265)

                # SAM3 Refinement
                with gr.Column(elem_classes="custom-card"):
                    gr.Markdown("### ✂️ Local Refinement (SAM3)", elem_classes="section-title")
                    gal_sam3_preview = gr.Gallery(label="SAM3 Visual Results", columns=5, height=220, object_fit="contain", preview=True)
                    file_sam3_editor = gr.File(label="Focus List Editor", file_count="multiple", type="filepath", height=100)

                # Final Decision
                with gr.Column(elem_classes="custom-card"):
                    with gr.Tabs():
                        with gr.TabItem("🤖 Final Diagnosis"):
                            md_final_res = gr.Markdown("### *Waiting for results...*")
                        
                        with gr.TabItem("🛠️ Logic Debug"):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1):
                                    gr.Markdown("### 👁️ Visual Input", elem_classes="section-title-sm")
                                    img_logic_debug = gr.Image(label="Logic View", type="pil", elem_classes="logic-view-img", show_label=False, interactive=False)
                                with gr.Column(scale=1):
                                    gr.Markdown("### 🧠 Constructed Prompt", elem_classes="section-title-sm")
                                    md_p1_prompt = gr.Markdown(value="*Waiting...*", elem_classes="prompt-card")

                        with gr.TabItem("🏗️ Chain of Thought (Manual Intervene)"):
                            gr.Markdown("💡 **Phase 2 Only**: Edit Prompt and rerun just the final conclusion.")
                            txt_cot_edit = gr.TextArea(label="Instruction", lines=8, interactive=True)
                            btn_manual_rerun = gr.Button("🔄 Rerun Phase 2 Only", variant="secondary", elem_classes="nju-btn-primary")
                        
                        with gr.TabItem("📟 Live Logs"):
                            txt_logs_stream = gr.TextArea(elem_classes="log-box", lines=12, show_copy_button=True)

                # Export / Download
                with gr.Column(elem_classes="custom-card"):
                    gr.Markdown("### 📦 Export & Download", elem_classes="section-title")
                    cb_export_docx = gr.Checkbox(
                        label="Include Word report (.docx) inside ZIP (requires python-docx)",
                        value=True,
                        interactive=True,
                    )
                    # Prefer 1-click download if Gradio supports DownloadButton; otherwise fallback to File output.
                    if hasattr(gr, "DownloadButton"):
                        try:
                            btn_export_zip = gr.DownloadButton(
                                "⬇️ Download Full Run (ZIP)",
                                variant="primary",
                                size="lg",
                                elem_classes="nju-btn-primary",
                            )
                        except TypeError:
                            btn_export_zip = gr.DownloadButton("⬇️ Download Full Run (ZIP)")
                        file_export_zip = None
                    else:
                        btn_export_zip = gr.Button(
                            "⬇️ Generate Export (ZIP)",
                            variant="primary",
                            size="lg",
                            elem_classes="nju-btn-primary",
                        )
                        file_export_zip = gr.File(label="Export ZIP", interactive=False, type="filepath")
                    txt_export_status = gr.Textbox(label="Export Status", interactive=False)

        # --- States ---
        state_opts_val = gr.State(value={})
        txt_hidden_t_type = gr.Textbox(visible=False)
        state_redbox_pil = gr.State()
        state_crops_paths = gr.State()

        # --- Event Bindings ---
        
        # 1. Initialize
        btn_init.click(
            lambda pm, pl, pg, ps, pd, gpu, mt, vlm, gu, dn: global_sys.initialize(pm, pl, pg, ps, pd, gpu, mt, vlm, gu, dn), 
            inputs=[p_model, p_ckpt, p_graph, p_sam, p_data, dd_gpu, dd_model_type, cb_vllm, sl_gpu_util, dd_dataset], 
            outputs=status_box
        )
        
        # 2. Update Category List when Dataset Changes
        def update_cat_list(ds_name):
            if ds_name == "mvtec":
                return gr.update(choices=MVTEC_CLASSES, value=MVTEC_CLASSES[0])
            else:
                return gr.update(choices=VISA_CLASSES, value=VISA_CLASSES[0])
        
        dd_dataset.change(update_cat_list, dd_dataset, dd_cat)

        # 3. Load Dataset & Reset Subclass/Logic Dropdowns
        def on_cat_load(c):
            msg = global_sys.data_manager.load_subclass(c)
            # FORCE RESET of dropdowns to prevent cross-dataset sticky values
            subs = ["All"] + global_sys.data_manager.subfolder_choices
            logics = ["All"] + global_sys.data_manager.task_type_choices
            return gr.update(choices=subs, value="All"), gr.update(choices=logics, value="All"), msg

        btn_load.click(on_cat_load, dd_cat, [dd_sub_type, dd_logic, status_box])

        # 4. Filter Samples
        def on_filter_update(s_f, l_f, search_t):
            if not global_sys.data_manager: return gr.update(choices=[])
            choices = global_sys.data_manager.filter_samples(
                subfolder=None if s_f == "All" else s_f,
                task_type=None if l_f == "All" else l_f,
                search_query=search_t
            )
            return gr.update(choices=choices)

        for trigger in [dd_sub_type, dd_logic, txt_search_case]:
            trigger.change(on_filter_update, [dd_sub_type, dd_logic, txt_search_case], dd_samples_list)

        # 5. Select Case & Update Options State
        def on_select_case_id(idx_str):
            if not idx_str: return [None]*6
            idx = int(idx_str.split("|")[0].strip())
            img, q, t, gt, opts = global_sys.data_manager.get_sample_by_idx(idx)
            # opts is passed to state_opts_val, and json.dumps to txt_opts_json
            return img, q, t, gt, opts, json.dumps(opts, indent=2, ensure_ascii=False)

        dd_samples_list.change(
            on_select_case_id, dd_samples_list, 
            [img_preview_in, txt_q_in, txt_hidden_t_type, txt_gt_out, state_opts_val, txt_opts_json]
        )

        # 6. Run Actions
        btn_run_main.click(
            search_runner_wrapper,
            inputs=[img_preview_in, txt_q_in, dd_cat, txt_hidden_t_type, state_opts_val, gr.State(None), gr.State(None)],
            outputs=[gal_rag_source, txt_rag_manual, img_hm_out, gal_sam3_preview, file_sam3_editor, txt_logs_stream, md_final_res, txt_cot_edit, state_redbox_pil, state_crops_paths, img_logic_debug, md_p1_prompt]
        )

        btn_rerun_all.click(
            search_runner_wrapper,
            inputs=[img_preview_in, txt_q_in, dd_cat, txt_hidden_t_type, state_opts_val, txt_rag_manual, gal_rag_source],
            outputs=[gal_rag_source, txt_rag_manual, img_hm_out, gal_sam3_preview, file_sam3_editor, txt_logs_stream, md_final_res, txt_cot_edit, state_redbox_pil, state_crops_paths, img_logic_debug, md_p1_prompt]
        )

        btn_manual_rerun.click(
            run_manual_inference_phase2,
            inputs=[gal_rag_source, txt_rag_manual, txt_cot_edit, state_redbox_pil, file_sam3_editor],
            outputs=[md_final_res, txt_logs_stream]
        )

        export_outputs = [file_export_zip, txt_export_status] if file_export_zip is not None else [btn_export_zip, txt_export_status]
        btn_export_zip.click(
            export_current_run_bundle,
            inputs=[
                img_preview_in,
                txt_q_in,
                txt_gt_out,
                dd_cat,
                txt_hidden_t_type,
                state_opts_val,
                gal_rag_source,
                txt_rag_manual,
                img_hm_out,
                gal_sam3_preview,
                file_sam3_editor,
                md_final_res,
                txt_cot_edit,
                img_logic_debug,
                md_p1_prompt,
                txt_logs_stream,
                cb_export_docx,
            ],
            outputs=export_outputs,
        )

    return demo

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch the KRagAD Gradio visualization UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/interface to bind.")
    parser.add_argument("--port", type=int, default=None, help="Port to bind. Defaults to the first free port from --start_port.")
    parser.add_argument("--start_port", type=int, default=7860, help="First port to try when --port is not set.")
    parser.add_argument("--max_queue", type=int, default=5, help="Gradio queue max size.")
    parser.add_argument("--show_api", action="store_true", help="Show Gradio API docs.")
    args = parser.parse_args()

    # Gradio 会用 httpx 对本地地址做连通性检查；若系统设置了 http_proxy/https_proxy，
    # 可能导致本地请求也走代理，从而出现 httpx.ConnectError: [Errno 111] Connection refused。
    # 这里显式设置 NO_PROXY/no_proxy，确保 localhost/loopback 不走代理。
    _no_proxy_hosts = "localhost,127.0.0.1,0.0.0.0"
    os.environ["NO_PROXY"] = ",".join(
        [h for h in (os.environ.get("NO_PROXY", ""), _no_proxy_hosts) if h]
    )
    os.environ["no_proxy"] = ",".join(
        [h for h in (os.environ.get("no_proxy", ""), _no_proxy_hosts) if h]
    )

    port = args.port if args.port is not None else find_free_port(args.start_port)
    print(f"🚀 NJU Sim-Inspec App Launched at http://localhost:{port}")
    app = create_ui()
    app.queue(max_size=args.max_queue).launch(server_name=args.host, server_port=port, show_api=args.show_api)


if __name__ == "__main__":
    main()
