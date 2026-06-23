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
import re
import html
from datetime import datetime
from io import BytesIO
from collections import defaultdict
from PIL import Image

from glls import paths as glls_paths

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
GLLS_PROJECT_ROOT = glls_paths.project_root()
GLLS_DATA_ROOT = glls_paths.data_root()

def _default_existing_path(configured_path, project_relative_path):
    project_path = os.path.join(GLLS_PROJECT_ROOT, project_relative_path)
    return project_path if os.path.exists(project_path) else configured_path

DEFAULT_MMAD_ROOT = _default_existing_path(glls_paths.dataset_root(), "datasets/MMAD")
DEFAULT_QA_ROOT = _default_existing_path(glls_paths.qa_root(), "qa_collection")
DEFAULT_GRAPH_ROOT = glls_paths.graph_cache_root()
DEFAULT_SAM3_PATH = glls_paths.sam3_path()
DEFAULT_VLM_MODEL_PATH = glls_paths.vlm_model_path()


def _default_adaptclip_checkpoint(dataset_name):
    domain = "visa" if str(dataset_name).lower() == "visa" else "mvtec"
    return os.path.join(glls_paths.adaptclip_root(), "checkpoints", f"{domain}_epoch_15.pth")


def _resolve_adaptclip_checkpoint(path_value, dataset_name):
    if not path_value:
        return _default_adaptclip_checkpoint(dataset_name)
    if os.path.isdir(path_value):
        domain = "visa" if str(dataset_name).lower() == "visa" else "mvtec"
        candidates = [
            os.path.join(path_value, "checkpoints", f"{domain}_epoch_15.pth"),
            os.path.join(path_value, f"{domain}_epoch_15.pth"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return path_value


DEFAULT_ADAPTCLIP_CHECKPOINT_PATH = _default_adaptclip_checkpoint("mvtec")


def _frontend_localizer_name(dataset_name, k_shot):
    dataset_key = str(dataset_name or "").strip().lower()
    shot = int(k_shot or 0)
    if dataset_key in {"mvtec", "visa"} and shot == 1:
        return "abound"
    return "adaptclip"

# Try importing vLLM
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
    print("vLLM library detected.")
except ImportError:
    HAS_VLLM = False
    print("vLLM not found. Acceleration disabled.")

# Import Project Modules
try:
    from glls.models.mcts_sam import MCTSQuestionSample
    from glls.models.localizer import ABounD_Localizer, AdaptCLIP_Localizer
    from glls.qa_trace import binary_anomaly_decision_override, summarize_method_participation
    from glls.rag.agent import SimInspecAgent
    from glls.seg.sam3_engine import Sam3Engine
    # Import Loaders
    from glls.data.dataset_loader import DSMVTecDatasetLoader, VisaDatasetLoader 
except ImportError as e:
    print(f"Project module import failed: {e}")
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

def _coerce_options(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_normal_support_images(dataset_root, dataset_name, category, limit):
    if int(limit or 0) <= 0:
        return []

    dataset_key = str(dataset_name or "").strip().lower()
    category = str(category)
    if dataset_key == "mvtec":
        candidate_dirs = [
            os.path.join(dataset_root, category, "image", "good"),
            os.path.join(dataset_root, category, "train", "good"),
            os.path.join(dataset_root, category, "test", "good"),
        ]
    else:
        candidate_dirs = [
            os.path.join(dataset_root, category, "train", "good"),
            os.path.join(dataset_root, category, "test", "good"),
            os.path.join(dataset_root, category, "image", "good"),
        ]

    images = []
    for directory in candidate_dirs:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isfile(path) and os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS:
                images.append(path)
        if images:
            break
    return images[: int(limit)]


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

def patch_starlette_template_response_compat():
    """Keep Gradio 4.44 working with newer Starlette TemplateResponse signatures."""
    try:
        import inspect
        from starlette.templating import Jinja2Templates
    except Exception:
        return

    current = getattr(Jinja2Templates, "TemplateResponse", None)
    if getattr(current, "_glls_compat_patched", False):
        return
    try:
        params = list(inspect.signature(current).parameters)
    except (TypeError, ValueError):
        return
    if len(params) < 3 or params[1] != "request":
        return

    original = current

    def compat_template_response(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.pop("context", None)
            remaining = args[2:]
            if context is None:
                context = {}
            request = kwargs.pop("request", None)
            if request is None and isinstance(context, dict):
                request = context.get("request")
            return original(self, request, name, context, *remaining, **kwargs)
        return original(self, *args, **kwargs)

    compat_template_response._glls_compat_patched = True
    Jinja2Templates.TemplateResponse = compat_template_response

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
        doc.add_heading("GLLS QA Method Run Report", level=0)
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
        return None, "No completed run to export yet."
    options = _coerce_options(options)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_iso = datetime.now().isoformat(timespec="seconds")
    export_root = tempfile.mkdtemp(prefix="glls_qa_export_")
    zip_path = os.path.join(export_root, f"glls_qa_run_{timestamp}.zip")
    top_dir = f"glls_qa_run_{timestamp}"

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
        "options": options,
        "has_docx": False,
    }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _zip_write_text(
            zf,
            f"{top_dir}/README.txt",
            "This ZIP contains one GLLS QA method run export.\n"
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
    return zip_path, f"Export ready: {os.path.basename(zip_path)}{docx_note}"

# ==========================================
# 2. Dataset Manager (INTELLIGENT PATH FIX)
# ==========================================
class DatasetManager:
    def __init__(self, root_path, qa_root_path=None, dataset_name="mvtec"):
        self.root_path = root_path
        self.qa_root_path = qa_root_path
        self.dataset_name = dataset_name
        self.loader = None
        self.loader_data_root = None
        self.loader_qa_root = None
        self.all_samples = []          
        self.subfolder_choices = []    
        self.task_type_choices = []    

    def _target_folder(self):
        dataset_folder_map = {
            "visa": "VisA",
            "mvtec": "DS-MVTec",
        }
        return dataset_folder_map.get(str(self.dataset_name).lower())

    def _resolve_dataset_root(self):
        raw_root = os.path.normpath(self.root_path)
        target_folder = self._target_folder()
        if os.path.basename(raw_root) == target_folder:
            return raw_root
        return os.path.join(raw_root, target_folder)

    def _resolve_qa_root(self, final_data_root):
        target_folder = self._target_folder()
        raw_qa_root = os.path.normpath(self.qa_root_path or "")
        if not raw_qa_root:
            return final_data_root

        if os.path.basename(raw_qa_root) == target_folder:
            return raw_qa_root

        dataset_specific_qa = os.path.join(raw_qa_root, target_folder)
        if os.path.exists(dataset_specific_qa):
            return dataset_specific_qa

        return raw_qa_root
        
    def load_subclass(self, category):
        """Loads category data with auto-path correction."""
        
        # --- 1. Intelligent Path Resolution ---
        target_folder = self._target_folder()

        if not target_folder:
            return f"Unknown dataset type: {self.dataset_name}"

        final_root = self._resolve_dataset_root()
        final_qa_root = self._resolve_qa_root(final_root)
        if not os.path.exists(final_root):
            return f"Dataset path not found: {final_root}. Please check Dataset Root."
        if not os.path.exists(final_qa_root):
            return f"QA path not found: {final_qa_root}. Please check QA Collection Root."

        # --- 2. Loader Initialization ---
        loader_needs_init = (
            self.loader is None or
            self.loader_data_root != final_root or
            self.loader_qa_root != final_qa_root
        )
        if loader_needs_init:
            if self.dataset_name.lower() == "visa":
                print(f"Initializing VisaDatasetLoader | data={final_root} | qa={final_qa_root}")
                self.loader = VisaDatasetLoader(root_path=final_root, qa_root_path=final_qa_root)
            else:
                print(f"Initializing DSMVTecDatasetLoader | data={final_root} | qa={final_qa_root}")
                self.loader = DSMVTecDatasetLoader(root_path=final_root, qa_root_path=final_qa_root)
            self.loader_data_root = final_root
            self.loader_qa_root = final_qa_root
            
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
            
            return (
                f"[{self.dataset_name.upper()}] {category} loaded: {len(self.all_samples)} QA rows"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Load failed: {str(e)}"

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
        
        print(f"Loading VLM: {self.model_type} | vLLM enabled: {self.use_vllm}")

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
                print("vLLM engine ready.")
                return
            except Exception as e:
                print(f"vLLM init failed: {e}")
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
        self.current_localizer = None

    def initialize(
        self,
        model_path,
        ckpt_path,
        graph_root,
        sam_path,
        dataset_root,
        qa_root,
        gpu_id,
        model_type,
        use_vllm,
        gpu_util,
        dataset_name,
        k_shot=1,
    ):
        try:
            k_shot = int(k_shot or 0)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            print(f"System Config: physical_gpu={gpu_id} | logical_device={device} | dataset={dataset_name}")

            # Initialize DatasetManager with the raw path; it handles resolution internally
            self.data_manager = DatasetManager(dataset_root, qa_root, dataset_name)
            localizer_name = _frontend_localizer_name(dataset_name, k_shot)
            if localizer_name == "abound":
                resolved_ckpt_path = glls_paths.abound_model_path()
                resolved_save_path = glls_paths.abound_save_path()
                localizer_reload_key = f"{resolved_ckpt_path}|{resolved_save_path}"
            else:
                resolved_ckpt_path = _resolve_adaptclip_checkpoint(ckpt_path, dataset_name)
                resolved_save_path = ""
                localizer_reload_key = resolved_ckpt_path
            
            # Re-init VLM if params changed
            if self.vlm is None or self.vlm.model_type != model_type or self.vlm.model_path != model_path or self.vlm.use_vllm != use_vllm:
                if self.vlm: 
                    print("Reloading VLM and freeing memory.")
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
                self.current_localizer != localizer_name or
                self.current_ckpt != localizer_reload_key
            )

            if localizer_needs_reload:
                print(f"(Re)Loading {localizer_name} Localizer for dataset: {dataset_name}...")
                if self.localizer:
                    del self.localizer
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                class Args: pass
                self.args = Args()
                self.args.dataset = dataset_name 
                self.args.dataset_root = self.data_manager._resolve_dataset_root()
                self.args.image_size = 518
                self.args.checkpoint_path = resolved_ckpt_path
                self.args.save_path = resolved_save_path
                self.args.k_shot = k_shot
                self.args.localizer = localizer_name
                
                # Default params
                self.args.features_list = [6, 12, 18, 24]
                self.args.num_visual_finetune_layers = 12
                self.args.depth = 7; self.args.n_ctx = 11; self.args.spe = 4
                self.args.w0 = 0.15; self.args.w1 = 0.35; self.args.w2 = 0.35; self.args.w3 = 0.15
                
                if localizer_name == "abound":
                    self.localizer = ABounD_Localizer(self.args, device=device)
                else:
                    self.localizer = AdaptCLIP_Localizer(self.args, device=device)
                
                self.current_dataset = dataset_name
                self.current_ckpt = localizer_reload_key
                self.current_localizer = localizer_name
            
            if self.args is not None:
                self.args.k_shot = k_shot
            if self.localizer is not None:
                self.localizer.k_shot = k_shot

            if not self.rag_agent:
                self.rag_agent = SimInspecAgent(None, graph_root, k_shot=1)
            
            if not self.sam_engine and os.path.exists(sam_path): 
                print(f"Loading SAM3 to {device}...")
                self.sam_engine = Sam3Engine(sam_path, device=device)
            
            self.is_initialized = True
            vllm_status = f"ON (Util: {gpu_util})" if (use_vllm and self.vlm.use_vllm) else "OFF"
            shot_mode = f"{k_shot}-shot ({'zero-shot' if k_shot <= 0 else 'few-shot'})"
            localizer_label = "ABounD" if localizer_name == "abound" else "AdaptCLIP"
            return (
                f"System ready | GPU:{gpu_id} | Model:{model_type} | vLLM:{vllm_status} | "
                f"Dataset:{dataset_name} | Localizer:{localizer_label} {shot_mode} | QA:curated"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Init failed: {str(e)}"

global_sys = GlobalSystem()

class StreamableMCTS(MCTSQuestionSample):
    def __init__(self, row, args, inference_engine, localizer, rag_agent, rag_blocks, sam_engine, update_callback):
        super().__init__(row, args, inference_engine, localizer, rag_agent=rag_agent, rag_blocks=rag_blocks, sam_engine=sam_engine)
        self.update_callback = update_callback 

    async def get_anomaly_heatmap(self, image):
        heatmap, obj_name = await super().get_anomaly_heatmap(image)
        
        if heatmap is None or np.sum(heatmap) == 0:
             await self.update_callback("log", "Warning: localizer returned empty heatmap (check dataset config).")
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
            await self.update_callback("log", f"MCTS depth {node.state['depth']} searching; score={score:.2f}")
        return score

# ==========================================
# 5. Async Logic Stream
# ==========================================

def _parse_answer_key(response):
    response = _as_text(response)
    match = re.search(r"The correct answer is\s*\(?([A-D])\)?", response, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    cands = re.findall(r"\b([A-D])\b", response)
    return cands[-1].upper() if cands else ""

def _compact_json(value, max_chars=900):
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = _as_text(value)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n..."
    return text

def _short_text(value, max_chars=180):
    text = _as_text(value).strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text

def _source_label(path_value):
    text = _as_text(path_value)
    if not text:
        return ""
    normalized = text.replace("\\", "/")
    for marker in ("/graph_index/", "/text_knowledge/"):
        if marker in normalized:
            return normalized.split(marker, 1)[1]
    return os.path.basename(normalized) or normalized

def _method_trace_markdown(debug_meta):
    debug_meta = debug_meta or {}
    participation = summarize_method_participation(debug_meta)
    rag_summary = participation.get("rag_source_backed_summary", {})
    mcts_budget = debug_meta.get("mcts_budget_config", {}) or {}
    mcts_search = debug_meta.get("mcts_search_summary", {}) or {}
    mcts_action_trace = debug_meta.get("mcts_action_trace", []) or []
    sam_attempts = debug_meta.get("sam_text_prompt_attempts", []) or []
    sam_hits = debug_meta.get("sam_text_prompt_hits", []) or []
    sam_scores = debug_meta.get("sam_mask_scores", []) or []
    rag_blocks = debug_meta.get("rag_block_provenance", []) or []
    graph_paths = []
    text_paths = []
    for block in rag_blocks:
        if not isinstance(block, dict):
            continue
        graph_path = block.get("graph_cache_path")
        source_path = block.get("source_json_path")
        if graph_path and graph_path not in graph_paths:
            graph_paths.append(graph_path)
        if source_path and source_path not in text_paths:
            text_paths.append(source_path)

    lines = [
        "#### Method Trace",
        f"- MCTS: `{participation.get('mcts_participation_status', 'unknown')}` | "
        f"budget `{_compact_json(mcts_budget, 220)}` | actions `{len(mcts_action_trace)}`",
        f"- Region evidence: proposals `{debug_meta.get('region_proposal_count', 0)}` | "
        f"crops `{debug_meta.get('crop_count', 0)}` | prompt-visible `{debug_meta.get('prompt_visible_crop_count', debug_meta.get('crop_count', 0))}`",
        f"- Heatmap: peak `{debug_meta.get('heatmap_peak_score', 0)}` | threshold `{debug_meta.get('anomaly_threshold', 0)}` | source `{debug_meta.get('threshold_source', 'unknown')}`",
        f"- SAM3: `{participation.get('sam_participation_status', 'unknown')}` | "
        f"text attempts `{len(sam_attempts)}` | hits `{len(sam_hits)}` | mask audits `{len(sam_scores)}`",
        f"- PVLA/RAG: blocks `{rag_summary.get('block_count', 0)}` | graph sources `{rag_summary.get('graph_cache_path_count', 0)}` | "
        f"text JSON sources `{rag_summary.get('source_json_path_count', 0)}` | visual cutouts `{rag_summary.get('visual_reference_source_backed_count', 0)}`",
    ]
    if mcts_search:
        lines.append(f"- Search summary: `{_compact_json(mcts_search, 280)}`")
    if graph_paths:
        lines.append(f"- Graph cache: `{_source_label(graph_paths[0])}`" + (f" plus {len(graph_paths) - 1} more" if len(graph_paths) > 1 else ""))
    if text_paths:
        lines.append(f"- Text knowledge: `{_source_label(text_paths[0])}`" + (f" plus {len(text_paths) - 1} more" if len(text_paths) > 1 else ""))
    return "\n".join(lines)

def _first_items(values, limit=4):
    items = []
    for value in values or []:
        if value not in items:
            items.append(value)
        if len(items) >= limit:
            break
    return items

def _html(value):
    return html.escape(_as_text(value), quote=True)

def _display_status(value):
    return _as_text(value).replace("_", " ")

def _top_count_text(counts, limit=4):
    if not isinstance(counts, dict) or not counts:
        return "none"
    pairs = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return ", ".join(f"{key} x{value}" for key, value in pairs)

def _metric_html(label, value, tone=""):
    tone_class = f" {tone}" if tone else ""
    return (
        f'<div class="method-kpi{tone_class}">'
        f"<span>{_html(label)}</span><strong>{_html(value)}</strong>"
        "</div>"
    )

def _item_list_html(items, empty_text):
    rows = [f"<li>{item}</li>" for item in items if item]
    if not rows:
        rows = [f"<li>{_html(empty_text)}</li>"]
    return "<ul>" + "".join(rows) + "</ul>"

def _method_process_placeholder():
    stages = [
        ("01", "Phase-1 Global Logic", "Original image + QA are inspected first to form a global structural report."),
        ("02", "Small Localizer + MCTS", "AdaptCLIP heatmap proposals drive MCTS region search and crop selection."),
        ("03", "SAM3 Structural Gate", "Category prompts and mask quality checks refine reliable structural cutouts."),
        ("04", "PVLA / RAG Recall", "Graph cache, text knowledge, and visual references are retrieved with source provenance."),
        ("05", "Phase-2 Fusion", "Global report, local crops, SAM3 evidence, and PVLA blocks are fused for the final answer."),
    ]
    cards = []
    for step, title, body in stages:
        cards.append(
            '<article class="method-stage muted">'
            f'<div class="stage-head"><span>{step}</span><h4>{_html(title)}</h4></div>'
            f"<p>{_html(body)}</p>"
            "</article>"
        )
    return (
        '<div class="method-process-card">'
        '<div class="process-head"><div><p class="eyebrow">Method playback</p>'
        '<h3>Run a QA row to replay the GLLS evidence flow</h3></div>'
        '<span class="process-badge">Waiting</span></div>'
        '<div class="method-stage-grid">' + "".join(cards) + "</div>"
        "</div>"
    )

def _method_process_html(debug_meta, final_prompt=""):
    debug_meta = debug_meta or {}
    participation = summarize_method_participation(debug_meta)
    mcts_budget = debug_meta.get("mcts_budget_config", {}) or {}
    mcts_search = debug_meta.get("mcts_search_summary", {}) or {}
    crop_audit = debug_meta.get("crop_evidence_audit", []) or []
    sam_attempts = debug_meta.get("sam_text_prompt_attempts", []) or []
    sam_prompt_audit = debug_meta.get("sam_prompt_selection_audit", []) or []
    sam_hits = debug_meta.get("sam_text_prompt_hits", []) or []
    sam_scores = debug_meta.get("sam_mask_scores", []) or []
    rag_blocks = debug_meta.get("rag_block_provenance", []) or []
    rag_summary = participation.get("rag_source_backed_summary", {}) or {}

    crop_items = []
    for item in crop_audit[:4]:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("source") or "crop"
        source = item.get("source", "unknown")
        reason = item.get("reason") or item.get("status") or ""
        score = item.get("heatmap_score", item.get("sam_score", ""))
        score_text = f", score={float(score):.3f}" if isinstance(score, (int, float)) else ""
        crop_items.append(
            f"<b>{_html(label)}</b> from <code>{_html(source)}</code> "
            f"<span>({_html(reason)}{_html(score_text)})</span>"
        )

    prompt_items = []
    prompt_source = sam_prompt_audit if sam_prompt_audit else sam_attempts
    for item in prompt_source[:5]:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "prompt")
        text = item.get("text", "")
        status = item.get("status", "attempted")
        reason = item.get("reason") or item.get("selection_reason") or item.get("source", "")
        prompt_items.append(
            f"<code>{_html(_display_status(role))}</code> {_html(text or 'box/ROI prompt')} "
            f"<span>-> {_html(_display_status(status))}{(' / ' + _html(_display_status(reason))) if reason else ''}</span>"
        )

    rag_regions = []
    graph_sources = []
    text_sources = []
    for block in rag_blocks:
        if not isinstance(block, dict):
            continue
        region = block.get("region")
        if region:
            rag_regions.append(str(region))
        graph_path = block.get("graph_cache_path")
        if graph_path:
            graph_sources.append(_source_label(graph_path))
        text_path = block.get("source_json_path")
        if text_path:
            text_sources.append(_source_label(text_path))

    phase_report = _as_text(debug_meta.get("phase_1_result")).strip() or "No Phase-1 report recorded."
    prompt_crop_labels = debug_meta.get("prompt_crop_labels", []) or []
    final_prompt_note = "built and sent" if final_prompt else "not generated yet"
    task_policy = debug_meta.get("task_policy", {}) or {}
    if isinstance(task_policy, dict):
        policy_bits = []
        for key, label in [
            ("use_local_anomaly_stream", "local anomaly stream"),
            ("use_mcts_search", "MCTS search"),
            ("use_sam3_local_refinement", "SAM3 refinement"),
            ("show_phase1_report_in_final_prompt", "Phase-1 report in Phase-2"),
            ("show_rag_blocks_in_final_prompt", "PVLA/RAG in Phase-2"),
        ]:
            if task_policy.get(key):
                policy_bits.append(label)
        phase1_role = _display_status(task_policy.get("phase1_logic_role") or "standard")
        policy_text = f"{', '.join(policy_bits) or 'global-only'}; phase1 role={phase1_role}"
    else:
        policy_text = _short_text(task_policy, 160) or "standard"

    mcts_actions = len(debug_meta.get("mcts_action_trace", []) or [])
    if isinstance(mcts_budget, dict) and mcts_budget:
        mcts_budget_text = (
            f"{mcts_budget.get('n_simulations', '?')} sims, depth "
            f"{mcts_budget.get('max_depth', '?')}, actions {mcts_budget.get('action_count', '?')}"
        )
    else:
        mcts_budget_text = _compact_json(mcts_budget, 120) or "not recorded"
    heatmap_score = debug_meta.get("heatmap_score", debug_meta.get("heatmap_peak_score", 0))
    threshold = debug_meta.get("threshold_used", debug_meta.get("anomaly_threshold", 0))
    selected_regions = []
    prompt_rag_audit = debug_meta.get("prompt_rag_selection_audit", {}) or {}
    if isinstance(prompt_rag_audit, dict):
        selected_regions.extend(prompt_rag_audit.get("selected_regions", []) or [])
    selected_regions.extend(rag_regions)
    sam_veto = debug_meta.get("sam3_normal_part_veto", {}) or {}
    if isinstance(sam_veto, dict) and sam_veto.get("status"):
        sam_gate = f"normal-part veto {_display_status(sam_veto.get('status'))}"
    elif int(debug_meta.get("sam_refined_crop_count", 0) or 0) > 0:
        sam_gate = f"{debug_meta.get('sam_refined_crop_count')} accepted refined crop(s)"
    elif sam_scores:
        sam_gate = "mask audited; no refined crop selected"
    else:
        sam_gate = "not needed for this task/category"

    summary = (
        _metric_html("MCTS", f"{_display_status(participation.get('mcts_participation_status', 'unknown'))} / {mcts_actions} actions", "signal")
        + _metric_html("SAM3", f"{_display_status(participation.get('sam_participation_status', 'unknown'))} / {len(sam_scores)} masks", "builder")
        + _metric_html("PVLA", f"{rag_summary.get('block_count', len(rag_blocks))} blocks", "pass")
        + _metric_html("Phase-2", final_prompt_note, "")
    )

    stage_1 = (
        '<article class="method-stage phase-global">'
        '<div class="stage-head"><span>01</span><h4>Phase-1 Global Logic Report</h4></div>'
        '<p>The first VLM pass reads the whole image and QA/options, then records a global structural report. '
        'It is kept as context for Phase-2 instead of replacing local evidence.</p>'
        + _item_list_html([
            f"Global report: <code>{_html(phase_report)}</code>",
            f"Policy gates: <code>{_html(policy_text)}</code>",
            f"Logic role: <code>{_html('global structural anchor' if not debug_meta.get('used_logic_engine') else 'logic engine active')}</code>",
        ], "No Phase-1 trace was recorded.")
        + "</article>"
    )

    stage_2 = (
        '<article class="method-stage phase-local">'
        '<div class="stage-head"><span>02</span><h4>Small Localizer + MCTS Crop Search</h4></div>'
        '<p>The small localizer supplies anomaly heatmap proposals. MCTS searches those proposals and keeps only compact local evidence for the final prompt.</p>'
        + _item_list_html([
            f"Localizer: <code>{_html(debug_meta.get('threshold_source', 'unknown'))}</code>, score <code>{_html(heatmap_score)}</code>, threshold <code>{_html(threshold)}</code>",
            f"Proposals: <code>{_html(debug_meta.get('region_proposal_count', 0))}</code>, selected crops <code>{_html(debug_meta.get('crop_count', 0))}</code>, prompt-visible <code>{_html(debug_meta.get('prompt_visible_crop_count', 0))}</code>",
            f"MCTS budget: <code>{_html(mcts_budget_text)}</code>; action trace <code>{_html(mcts_actions)}</code>",
            f"Top actions: <code>{_html(_top_count_text((mcts_search or {}).get('expanded_action_counts', {})))}</code>",
            *crop_items,
        ], "No local crop was selected for this question.")
        + "</article>"
    )

    stage_3 = (
        '<article class="method-stage phase-sam">'
        '<div class="stage-head"><span>03</span><h4>SAM3 Structural Cut and Logic Gate</h4></div>'
        '<p>SAM3 is used when structural segmentation can improve local evidence. Prompt candidates are audited, masks are quality checked, and accepted masks become crop evidence.</p>'
        + _item_list_html([
            f"SAM3 status: <code>{_html(_display_status(participation.get('sam_participation_status', 'unknown')))}</code>",
            f"Structural gate: <code>{_html(sam_gate)}</code>",
            f"Mask scores: <code>{_html(_compact_json(sam_scores[:6], 160))}</code>",
            f"Artifact rule: <code>{_html(_short_text(debug_meta.get('sam3_crop_artifact_rule', 'not recorded'), 140))}</code>",
            *prompt_items,
        ], "No SAM3 prompt or mask audit was needed.")
        + "</article>"
    )

    stage_4 = (
        '<article class="method-stage phase-pvla">'
        '<div class="stage-head"><span>04</span><h4>PVLA / RAG Multimodal Knowledge</h4></div>'
        '<p>PVLA keeps the graph-shaped hierarchy visible: region-level text knowledge, graph cache provenance, and normal/reference visual cutouts are recalled before Phase-2.</p>'
        + _item_list_html([
            f"Retrieved blocks: <code>{_html(rag_summary.get('block_count', len(rag_blocks)))}</code>, graph sources <code>{_html(rag_summary.get('graph_cache_path_count', len(set(graph_sources))))}</code>, text sources <code>{_html(rag_summary.get('source_json_path_count', len(set(text_sources))))}</code>",
            f"Visual references: <code>{_html(rag_summary.get('visual_reference_source_backed_count', 0))}</code> source-backed cutout(s)",
            f"Selected regions: <code>{_html(', '.join(_first_items(selected_regions, 6)) or 'none')}</code>",
            f"Graph caches: <code>{_html(', '.join(_first_items(graph_sources, 3)) or 'none')}</code>",
            f"Text knowledge: <code>{_html(', '.join(_first_items(text_sources, 3)) or 'none')}</code>",
        ], "No PVLA/RAG block was selected.")
        + "</article>"
    )

    stage_5 = (
        '<article class="method-stage phase-final">'
        '<div class="stage-head"><span>05</span><h4>Phase-2 Evidence Fusion</h4></div>'
        '<p>The final VLM answer sees the original image plus the selected method evidence, then emits a constrained multiple-choice answer.</p>'
        + _item_list_html([
            f"Prompt crops: <code>{_html(', '.join(prompt_crop_labels) or 'none')}</code>",
            f"PVLA blocks in prompt: <code>{_html(len(debug_meta.get('prompt_rag_text_blocks', []) or []))}</code>",
            f"Final prompt: <code>{_html(final_prompt_note)}</code>",
            f"Verification strategy: <code>{_html(debug_meta.get('verification_strategy', 'standard'))}</code>",
        ], "Phase-2 prompt has not been generated.")
        + "</article>"
    )

    return (
        '<div class="method-process-card">'
        '<div class="process-head"><div><p class="eyebrow">Method playback</p>'
        '<h3>GLLS evidence flow for this QA</h3>'
        '<p>Each stage below is rendered from the run trace, so it shows what the method actually searched, cut, recalled, and sent to Phase-2.</p>'
        '</div><span class="process-badge">Trace-backed</span></div>'
        '<div class="method-kpi-grid">' + summary + "</div>"
        '<div class="method-stage-grid">' + stage_1 + stage_2 + stage_3 + stage_4 + stage_5 + "</div>"
        "</div>"
    )

def _final_result_markdown(question, options, pred_key, gt_key, is_correct, final_response, debug_meta, ad_override):
    options = options or {}
    pred_text = options.get(pred_key, "") if pred_key else ""
    gt_text = options.get(gt_key, "") if gt_key else ""
    result_label = "Unknown"
    if is_correct is True:
        result_label = "Correct"
    elif is_correct is False:
        result_label = "Wrong"

    option_lines = []
    for key in sorted(options.keys()):
        marker = []
        if key == pred_key:
            marker.append("pred")
        if key == gt_key:
            marker.append("gt")
        suffix = f" ({', '.join(marker)})" if marker else ""
        option_lines.append(f"- **{key}.** {options[key]}{suffix}")

    gate_line = "not applicable"
    if ad_override:
        gate_line = (
            f"{'applied' if ad_override.get('applied') else 'checked'} | "
            f"{ad_override.get('reason', 'no reason')} | "
            f"{ad_override.get('original_pred_key')} -> {ad_override.get('override_pred_key')}"
        )

    return "\n".join([
        "### Question Result",
        f"**Question:** {question}",
        "",
        "| Prediction | Ground Truth | Result | Binary AD Gate |",
        "| --- | --- | --- | --- |",
        f"| **{pred_key or 'N/A'}** {pred_text} | **{gt_key or 'N/A'}** {gt_text} | **{result_label}** | {gate_line} |",
        "",
        "#### Options",
        "\n".join(option_lines) if option_lines else "No options loaded.",
        "",
        _method_trace_markdown(debug_meta),
        "",
        "#### Model Response",
        _as_text(final_response).strip() or "No response.",
    ])

async def run_analysis_stream(image, question, subclass, task_type, options, ground_truth=None, override_rag_text=None, override_rag_files=None):
    if not global_sys.is_initialized:
        yield None, "", None, None, None, "System not initialized.", "", "", None, None, None, "", _method_process_placeholder()
        return
    if image is None or not question:
        yield None, "", None, None, None, "No QA sample selected.", "", "", None, None, None, "", _method_process_placeholder()
        return
    
    # Debug Options Passing
    if not options:
        print("[DEBUG] No options received in run_analysis_stream.")
    else:
        print(f"[DEBUG] Options received: {len(options)} keys")

    # --- RAG Loading Logic (Updated for Smart Splitting) ---
    rag_blocks = []
    
    if override_rag_text or (override_rag_files is not None and len(override_rag_files) > 0):
        print("Using manual RAG context overrides (auto-splitting global/local).")
        
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
    
    state = {"heatmap": None, "log": "Starting GLLS method trace."}
    if override_rag_text:
        state["log"] += "\nNOTE: Running with edited knowledge blocks."
    if global_sys.vlm.use_vllm:
        state["log"] += "\nvLLM acceleration: active."

    configure_support = getattr(global_sys.localizer, "configure_support", None)
    if callable(configure_support):
        dataset_root = global_sys.data_manager._resolve_dataset_root()
        support_paths = find_normal_support_images(
            dataset_root,
            global_sys.data_manager.dataset_name,
            subclass,
            getattr(global_sys.args, "k_shot", 1),
        )
        configure_support(subclass, support_paths)
        state["log"] += f"\nAdaptCLIP support: {len(support_paths)} normal reference image(s)."

    # Yield 1: Init
    yield rag_file_paths, rag_text_content, None, None, None, state["log"], "", "", None, None, None, "", _method_process_placeholder()

    async def callback(key, value):
        if key == "heatmap": state["heatmap"] = value
        if key == "log": state["log"] += f"\n{value}"

    row = {'image': image.convert("RGB"), 'question': question, 'options': options, 'category': subclass, 'type': task_type}
    try:
        mcts = StreamableMCTS(row, global_sys.args, global_sys.vlm, global_sys.localizer, global_sys.rag_agent, rag_blocks, global_sys.sam_engine, callback)
        mcts_task = asyncio.create_task(mcts.process())
        
        while not mcts_task.done():
            await asyncio.sleep(0.1)
            yield rag_file_paths, rag_text_content, state["heatmap"], None, None, state["log"], "", "", None, None, None, "", _method_process_placeholder()
        
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
            formatted_prompt = formatted_prompt.replace("=== PHASE 1: GLOBAL INSPECTION ===", "### PHASE 1: GLOBAL INSPECTION")
            formatted_prompt = formatted_prompt.replace("STEP 1:", "\n**STEP 1:**").replace("STEP 2:", "\n**STEP 2:**").replace("STEP 3:", "\n**STEP 3:**")
            formatted_prompt = formatted_prompt.replace("INSTRUCTION:", "\n#### INSTRUCTION:")
            formatted_prompt = formatted_prompt.replace("[SPECIAL INSPECTION VIEW]", "**[SPECIAL INSPECTION VIEW]**")
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
        
        debug_meta = result.get("debug_metadata", {}) or {}
        state["log"] += "\nGenerating final diagnosis."
        final_response = global_sys.vlm.generate(final_content)
        pred_key = _parse_answer_key(final_response)
        gt_key = _as_text(ground_truth).strip()
        ad_override = binary_anomaly_decision_override(task_type, options, debug_meta, pred_key) if pred_key else None
        if ad_override:
            pred_key = ad_override.get("override_pred_key", pred_key)
            debug_meta["binary_anomaly_decision_override"] = ad_override
        is_correct = (pred_key == gt_key) if pred_key and gt_key else None
        final_result_md = _final_result_markdown(
            question,
            options,
            pred_key,
            gt_key,
            is_correct,
            final_response,
            debug_meta,
            ad_override,
        )
        method_process_md = _method_process_html(debug_meta, final_prompt)
        state["log"] += (
            "\nAnalysis complete."
            f"\nPrediction: {pred_key or 'N/A'} | GT: {gt_key or 'N/A'} | Result: "
            f"{'Correct' if is_correct is True else 'Wrong' if is_correct is False else 'Unknown'}"
        )
        
        yield (
            rag_file_paths,     
            rag_text_content,   
            state["heatmap"],   
            gallery_items,      
            crop_file_paths,    
            state["log"],       
            final_result_md,     
            final_prompt,       
            result.get("red_box_image"), 
            crop_file_paths,
            debug_view_img,
            formatted_prompt,
            method_process_md
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield rag_file_paths, rag_text_content, state["heatmap"], None, None, f"Error: {str(e)}", "", "", None, None, None, "", f"<div class='method-process-card'><h3>Method run failed</h3><p>{_html(str(e))}</p></div>"

# Wrapper for Gradio Generator
def search_runner_wrapper(img, q, sub, task, opts, gt=None, man_txt=None, man_files=None):
    # Verify opts before running
    opts = _coerce_options(opts)
    if not opts:
        print("[WRAPPER] Options are None or invalid, resetting to empty dict.")
        
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    gen = run_analysis_stream(img, q, sub, task, opts, gt, man_txt, man_files)
    try:
        while True: yield loop.run_until_complete(gen.__anext__())
    except StopAsyncIteration: pass
    finally: loop.close()

# ==========================================
# 6. Gradio UI
# ==========================================

CSS = """
:root {
    --canvas: #F5F7FB;
    --panel: #FFFFFF;
    --panel-raised: #FBFCFE;
    --rule: #DCE2EA;
    --ink: #1B2230;
    --muted: #5A6675;
    --signal: #3452C7;
    --signal-hover: #2A43A8;
    --signal-subtle: #EEF1FB;
    --builder: #B0640F;
    --builder-subtle: #F8EFE4;
    --pass: #0F7A4D;
    --pass-subtle: #EAF6F0;
    --danger: #C0392B;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 2px rgba(27, 34, 48, 0.05), 0 1px 3px rgba(27, 34, 48, 0.04);
    --shadow-md: 0 4px 12px rgba(27, 34, 48, 0.08), 0 2px 4px rgba(27, 34, 48, 0.05);
}
html, body, .gradio-container {
    background:
        linear-gradient(90deg, rgba(220, 226, 234, 0.62) 1px, transparent 1px) 0 0 / 72px 72px,
        var(--canvas) !important;
    color: var(--ink);
    font-family: Inter, "IBM Plex Sans", "Segoe UI", "Microsoft YaHei UI", system-ui, sans-serif;
}
.gradio-container {
    max-width: none !important;
}
.gradio-container .contain {
    max-width: 1560px !important;
}
.app-topbar {
    position: sticky;
    top: 0;
    z-index: 20;
    margin: -16px -16px 0;
    padding: 14px 28px;
    border-bottom: 1px solid var(--rule);
    background: rgba(247, 248, 250, 0.94);
    backdrop-filter: blur(16px);
}
.brand-block {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-mark {
    width: 13px;
    min-height: 42px;
    border-radius: 999px;
    background: linear-gradient(var(--signal), var(--builder) 52%, var(--pass));
    box-shadow: inset 0 0 0 1px rgba(31, 35, 40, 0.16);
}
.brand-copy strong {
    display: block;
    font-family: Literata, "Source Serif 4", Georgia, "Microsoft YaHei", serif;
    font-size: 22px;
    line-height: 1.05;
}
.brand-copy small,
.hero-copy,
.section-copy,
.muted-note {
    color: var(--muted);
}
.status-chip textarea,
.status-chip input {
    min-height: 36px !important;
    border-radius: 999px !important;
    border: 1px solid var(--rule) !important;
    background: var(--panel) !important;
    color: var(--ink) !important;
    font-size: 0.86rem !important;
}
.hero-panel {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(260px, 360px);
    gap: 18px;
    align-items: end;
    margin: 18px 0 14px;
    padding: 18px 20px;
    border: 1px solid var(--rule);
    border-radius: var(--radius-lg);
    background: var(--panel);
    box-shadow: var(--shadow-sm);
}
.hero-panel h1 {
    margin: 4px 0 8px;
    color: var(--ink);
    font-family: Literata, "Source Serif 4", Georgia, "Microsoft YaHei", serif;
    font-size: clamp(1.9rem, 3.4vw, 3.1rem);
    line-height: 1.02;
    letter-spacing: 0;
}
.eyebrow {
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0;
    text-transform: uppercase;
}
.hero-metrics {
    display: grid;
    gap: 8px;
}
.hero-metrics span {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    padding: 8px 12px;
    border: 1px solid var(--rule);
    border-radius: var(--radius-md);
    background: var(--panel-raised);
}
.hero-metrics b {
    color: var(--signal);
}
.custom-card {
    background: var(--panel);
    border-radius: var(--radius-lg);
    padding: 18px;
    border: 1px solid var(--rule);
    box-shadow: var(--shadow-sm);
}
.custom-card.tight {
    padding: 14px;
}
.section-title {
    color: var(--ink);
    font-weight: 800;
    font-size: 1.08rem;
    border-left: 4px solid var(--signal);
    padding-left: 12px;
    margin-bottom: 8px;
}
.section-title-sm {
    color: var(--ink);
    font-weight: 800;
    font-size: 0.98rem;
    margin-bottom: 8px;
}
.section-copy {
    margin: 0 0 12px;
    font-size: 0.92rem;
}
.method-strip {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 8px;
}
.method-strip span {
    min-height: 42px;
    padding: 8px 10px;
    border: 1px solid var(--rule);
    border-radius: 999px;
    background: var(--panel-raised);
    color: var(--ink);
    font-size: 0.86rem;
    font-weight: 700;
    text-align: center;
}
.method-strip span:nth-child(1) { background: var(--signal-subtle); color: var(--signal); }
.method-strip span:nth-child(2) { background: var(--builder-subtle); color: var(--builder); }
.method-strip span:nth-child(3) { background: var(--pass-subtle); color: var(--pass); }
.method-strip span:nth-child(4) { background: #F2F4F7; }
.method-strip span:nth-child(5) { background: #F7F1EA; color: var(--builder); }
.desk-btn-primary {
    background: var(--signal) !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--signal-hover) !important;
    box-shadow: 0 1px 2px rgba(52, 82, 199, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
}
.desk-btn-primary:hover {
    background: var(--signal-hover) !important;
}
.desk-btn-secondary {
    background: var(--panel) !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
    border: 1px solid var(--rule) !important;
    border-radius: var(--radius-md) !important;
}
.gradio-container button {
    border-radius: var(--radius-md) !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    border-radius: var(--radius-sm) !important;
}
.gradio-container label,
.gradio-container .label-wrap span {
    color: var(--muted) !important;
    font-weight: 700 !important;
}
.desk-image-upload,
.evidence-image {
    border-radius: var(--radius-md);
    overflow: hidden;
}
.log-box textarea {
    background: #1B2230 !important;
    color: #B9F6D6 !important;
    font-family: "Cascadia Code", Consolas, monospace !important;
    border-radius: var(--radius-md) !important;
}
.prompt-card {
    background: var(--panel-raised);
    border: 1px solid var(--rule);
    border-radius: var(--radius-md);
    padding: 16px;
    min-height: 360px !important;
    max-height: 460px !important;
    overflow-y: auto;
    font-size: 0.95rem;
    line-height: 1.6;
}
.prompt-card h3 {
    color: var(--signal);
    border-bottom: 1px solid var(--rule);
    padding-bottom: 8px;
    margin-top: 10px;
}
.prompt-card strong {
    color: var(--ink);
    background-color: var(--signal-subtle);
    padding: 0 4px;
    border-radius: 4px;
}
.prompt-card h3 {
    font-family: Literata, "Source Serif 4", Georgia, "Microsoft YaHei", serif;
    font-size: 1.45rem;
}
.prompt-card h4 {
    margin: 16px 0 8px;
    color: var(--ink);
    border-top: 1px solid var(--rule);
    padding-top: 12px;
}
.prompt-card code {
    background: #EEF1FB;
    color: #1B2230;
    border: 1px solid #D8E0F2;
    border-radius: 4px;
    padding: 1px 4px;
}
.method-process-card {
    background: transparent;
    border: 0;
    padding: 0;
    font-size: 0.95rem;
    line-height: 1.55;
}
.method-process-card h3 {
    font-family: Literata, "Source Serif 4", Georgia, "Microsoft YaHei", serif;
    font-size: 1.45rem;
    margin: 0;
}
.method-process-card p {
    margin: 6px 0 0;
    color: var(--muted);
}
.process-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 18px;
    padding: 2px 2px 14px;
}
.process-badge {
    display: inline-flex;
    align-items: center;
    min-height: 30px;
    padding: 4px 10px;
    border: 1px solid #B7D7C8;
    border-radius: 999px;
    background: var(--pass-subtle);
    color: var(--pass);
    font-size: 0.82rem;
    font-weight: 800;
    white-space: nowrap;
}
.method-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 0 0 12px;
}
.method-kpi {
    min-height: 68px;
    border: 1px solid var(--rule);
    border-radius: var(--radius-md);
    background: var(--panel-raised);
    padding: 10px 12px;
}
.method-kpi span {
    display: block;
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 800;
}
.method-kpi strong {
    display: block;
    margin-top: 4px;
    color: var(--ink);
    font-size: 1rem;
    overflow-wrap: anywhere;
}
.method-kpi.signal strong { color: var(--signal); }
.method-kpi.builder strong { color: var(--builder); }
.method-kpi.pass strong { color: var(--pass); }
.method-stage-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}
.method-stage {
    border: 1px solid var(--rule);
    border-radius: var(--radius-md);
    background: var(--panel-raised);
    padding: 14px;
    min-height: 220px;
}
.method-stage:nth-child(5),
.method-stage.phase-final {
    grid-column: 1 / -1;
    min-height: auto;
}
.method-stage.muted {
    min-height: 150px;
}
.stage-head {
    display: flex;
    gap: 10px;
    align-items: center;
}
.stage-head span {
    display: inline-grid;
    place-items: center;
    width: 32px;
    height: 32px;
    border-radius: 999px;
    background: var(--signal-subtle);
    color: var(--signal);
    font-weight: 900;
    font-size: 0.82rem;
}
.phase-local .stage-head span { background: var(--builder-subtle); color: var(--builder); }
.phase-sam .stage-head span { background: var(--pass-subtle); color: var(--pass); }
.phase-pvla .stage-head span { background: #F2F4F7; color: var(--ink); }
.phase-final .stage-head span { background: #F7F1EA; color: var(--builder); }
.method-stage h4 {
    margin: 0;
    color: var(--ink);
    font-size: 1rem;
    line-height: 1.25;
}
.method-stage ul {
    margin: 10px 0 0;
    padding-left: 18px;
}
.method-stage li {
    margin: 6px 0;
    color: var(--ink);
    overflow-wrap: anywhere;
}
.method-stage li span {
    color: var(--muted);
}
.method-process-card code {
    background: #EEF1FB;
    color: #1B2230;
    border: 1px solid #D8E0F2;
    border-radius: 4px;
    padding: 1px 4px;
    overflow-wrap: anywhere;
}
.logic-view-img {
    height: 360px !important;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--rule);
    display: flex;
    justify-content: center;
    align-items: center;
    background: var(--panel-raised);
}
.hidden-artifact {
    display: none !important;
}
@media (max-width: 1100px) {
    .hero-panel {
        grid-template-columns: 1fr;
    }
    .method-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .method-kpi-grid,
    .method-stage-grid {
        grid-template-columns: 1fr;
    }
    .method-stage:nth-child(5),
    .method-stage.phase-final {
        grid-column: auto;
    }
}
"""

def create_ui():
    desk_theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate").set(
        block_radius="8px",
        button_primary_background_fill="#3452C7",
        button_primary_background_fill_hover="#2A43A8",
    )

    MVTEC_CLASSES = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", 
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    VISA_CLASSES = [
        "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", 
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    ]
    
    with gr.Blocks(css=CSS, theme=desk_theme, title="GLLS QA Method Desk") as demo:

        with gr.Row(elem_classes="app-topbar"):
            with gr.Column(scale=3):
                gr.HTML(
                    """
                    <div class="brand-block">
                      <div class="brand-mark"></div>
                      <div class="brand-copy">
                        <strong>GLLS QA Method Desk</strong>
                        <small>Global logic, local evidence, SAM3 refinement, and PVLA/RAG provenance.</small>
                      </div>
                    </div>
                    """
                )
            with gr.Column(scale=2, elem_classes="status-chip"):
                status_box = gr.Textbox(label="Runtime", value="Not initialized", interactive=False, container=False)

        gr.HTML(
            """
            <section class="hero-panel">
              <div>
                <div class="eyebrow">Method playback</div>
                <h1>GLLS Method Playback</h1>
                <p class="hero-copy">
                  Pick a QA sample and replay the two-stage reasoning path: Phase-1 global
                  report, small-model heatmap proposals, MCTS crop search, SAM3 structural
                  refinement, PVLA/RAG recall, and Phase-2 answer fusion.
                </p>
              </div>
              <div class="hero-metrics">
                <span><b>1</b><em>Initialize from local config</em></span>
                <span><b>2</b><em>Load dataset/category QA rows</em></span>
                <span><b>3</b><em>Run and export the evidence bundle</em></span>
              </div>
            </section>
            """
        )

        with gr.Accordion("Runtime setup (read from local_paths.sh by default)", open=False):
            gr.Markdown(
                "Set paths once in `scripts/dev/local_paths.sh`, then start this page with "
                "`bash scripts/run/run_visualize.sh`. These fields are only for overriding the active session."
            )
            with gr.Row():
                dd_model_type = gr.Dropdown(
                    ["qwen3-vl", "qwen2.5-vl", "llava-onevision"],
                    label="VLM architecture",
                    value="qwen3-vl",
                )
                dd_gpu = gr.Dropdown([str(i) for i in range(8)], value="0", label="GPU")
                dd_kshot = gr.Dropdown(
                    ["1", "0"],
                    value="1",
                    label="Localizer shots (1 = one-shot, 0 = zero-shot)",
                )
                cb_vllm = gr.Checkbox(label="Use vLLM if installed", value=False, interactive=True)
                sl_gpu_util = gr.Slider(0.3, 0.95, value=0.85, step=0.05, label="vLLM memory fraction")
            with gr.Row():
                p_model = gr.Textbox(DEFAULT_VLM_MODEL_PATH, label="VLM model folder")
                p_sam = gr.Textbox(DEFAULT_SAM3_PATH, label="SAM3 checkpoint")
            with gr.Row():
                p_data = gr.Textbox(DEFAULT_MMAD_ROOT, label="MMAD dataset root")
                p_qa = gr.Textbox(DEFAULT_QA_ROOT, label="Curated QA root")
            with gr.Row():
                p_ckpt = gr.Textbox(
                    DEFAULT_ADAPTCLIP_CHECKPOINT_PATH,
                    label="AdaptCLIP checkpoint (used when AdaptCLIP is selected)",
                )
                p_graph = gr.Textbox(DEFAULT_GRAPH_ROOT, label="PVLA graph cache root")
            btn_init = gr.Button("Initialize GLLS runtime", elem_classes="desk-btn-primary")

        with gr.Row():
            with gr.Column(scale=1, min_width=390):
                with gr.Column(elem_classes="custom-card"):
                    gr.Markdown("### QA Sample", elem_classes="section-title")
                    gr.Markdown("Choose a dataset/category, load QA rows, then select one question.", elem_classes="section-copy")
                    dd_dataset = gr.Dropdown(["mvtec", "visa"], label="Dataset", value="mvtec")
                    dd_cat = gr.Dropdown(MVTEC_CLASSES, label="Category", value="bottle", interactive=True)
                    btn_load = gr.Button("Load QA rows", size="sm", elem_classes="desk-btn-secondary")
                    with gr.Row():
                        dd_sub_type = gr.Dropdown(label="Defect folder", choices=["All"], value="All")
                        dd_logic = gr.Dropdown(label="Task", choices=["All"], value="All")
                    txt_search_case = gr.Textbox(placeholder="image name / question / task", label="Search")
                    dd_samples_list = gr.Dropdown(label="Question", choices=[], interactive=True)
                    img_preview_in = gr.Image(label="Input image", type="pil", height=300, elem_classes="desk-image-upload")
                    txt_q_in = gr.Textbox(label="Question", lines=2)
                    with gr.Accordion("Answer options", open=False):
                        txt_gt_out = gr.Textbox(label="Ground truth", interactive=False)
                        txt_opts_json = gr.Code(label="Options", language="json")
                    btn_run_main = gr.Button("Run GLLS on this question", variant="primary", size="lg", elem_classes="desk-btn-primary")

                with gr.Column(elem_classes="custom-card tight"):
                    gr.Markdown("### Method Streams", elem_classes="section-title")
                    gr.HTML(
                        """
                        <div class="method-strip">
                          <span>Phase-1 global</span>
                          <span>Small model + MCTS</span>
                          <span>SAM3 logic cut</span>
                          <span>PVLA/RAG recall</span>
                          <span>Phase-2 answer</span>
                        </div>
                        """
                    )

            with gr.Column(scale=2):
                with gr.Column(elem_classes="custom-card"):
                    gr.Markdown("### Method Process", elem_classes="section-title")
                    md_method_process = gr.HTML(_method_process_placeholder())

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="custom-card"):
                        gr.Markdown("### Global Heatmap", elem_classes="section-title")
                        img_hm_out = gr.Image(label="Heatmap evidence", type="pil", height=280, elem_classes="evidence-image")
                    with gr.Column(scale=1, elem_classes="custom-card"):
                        gr.Markdown("### SAM3 / Local Evidence", elem_classes="section-title")
                        gal_sam3_preview = gr.Gallery(
                            label="Selected local views",
                            columns=4,
                            height=280,
                            object_fit="contain",
                            preview=True,
                        )
                        file_sam3_editor = gr.File(
                            label="Selected focus files",
                            file_count="multiple",
                            type="filepath",
                            visible=False,
                        )

                with gr.Column(elem_classes="custom-card"):
                    gr.Markdown("### Source-backed PVLA Knowledge", elem_classes="section-title")
                    with gr.Tabs():
                        with gr.TabItem("Visual references"):
                            gal_rag_source = gr.Gallery(
                                label="Retrieved normal/reference cutouts",
                                show_label=True,
                                columns=4,
                                rows=1,
                                height=180,
                                object_fit="contain",
                                type="filepath",
                            )
                        with gr.TabItem("Text knowledge"):
                            txt_rag_manual = gr.TextArea(label="Retrieved knowledge blocks", lines=7, interactive=False)

                with gr.Column(elem_classes="custom-card"):
                    gr.Markdown("### Answer and Evidence Trace", elem_classes="section-title")
                    with gr.Tabs():
                        with gr.TabItem("Question result"):
                            md_final_res = gr.Markdown("### Waiting for a run")

                        with gr.TabItem("Prompt evidence"):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1):
                                    gr.Markdown("### Visual input", elem_classes="section-title-sm")
                                    img_logic_debug = gr.Image(label="Logic view", type="pil", elem_classes="logic-view-img", show_label=False, interactive=False)
                                with gr.Column(scale=1):
                                    gr.Markdown("### Phase-1 prompt", elem_classes="section-title-sm")
                                    md_p1_prompt = gr.Markdown(value="Waiting for a run.", elem_classes="prompt-card")
                            txt_cot_edit = gr.TextArea(label="Final prompt", lines=8, interactive=False)

                        with gr.TabItem("Run logs"):
                            txt_logs_stream = gr.TextArea(elem_classes="log-box", lines=12, show_copy_button=True)

                with gr.Column(elem_classes="custom-card"):
                    gr.Markdown("### Export", elem_classes="section-title")
                    cb_export_docx = gr.Checkbox(
                        label="Include Word report (.docx) inside ZIP",
                        value=True,
                        interactive=True,
                    )
                    if hasattr(gr, "DownloadButton"):
                        try:
                            btn_export_zip = gr.DownloadButton(
                                "Download full run bundle",
                                variant="primary",
                                size="lg",
                                elem_classes="desk-btn-primary",
                            )
                        except TypeError:
                            btn_export_zip = gr.DownloadButton("Download full run bundle")
                        file_export_zip = None
                    else:
                        btn_export_zip = gr.Button(
                            "Generate export bundle",
                            variant="primary",
                            size="lg",
                            elem_classes="desk-btn-primary",
                        )
                        file_export_zip = gr.File(label="Export ZIP", interactive=False, type="filepath")
                    txt_export_status = gr.Textbox(label="Export status", interactive=False)

        # --- States ---
        txt_hidden_t_type = gr.Textbox(visible=False)
        state_redbox_pil = gr.State()
        state_crops_paths = gr.State()

        # --- Event Bindings ---
        
        # 1. Initialize
        btn_init.click(
            lambda pm, pl, pg, ps, pd, pq, gpu, mt, vlm, gu, dn, ks: global_sys.initialize(pm, pl, pg, ps, pd, pq, gpu, mt, vlm, gu, dn, ks),
            inputs=[p_model, p_ckpt, p_graph, p_sam, p_data, p_qa, dd_gpu, dd_model_type, cb_vllm, sl_gpu_util, dd_dataset, dd_kshot],
            outputs=status_box,
            api_name=False,
        )
        
        # 2. Update Category List when Dataset Changes
        def update_cat_list(ds_name):
            if ds_name == "mvtec":
                return (
                    gr.update(choices=MVTEC_CLASSES, value=MVTEC_CLASSES[0]),
                    gr.update(value=_default_adaptclip_checkpoint("mvtec")),
                )
            else:
                return (
                    gr.update(choices=VISA_CLASSES, value=VISA_CLASSES[0]),
                    gr.update(value=_default_adaptclip_checkpoint("visa")),
                )
        
        dd_dataset.change(update_cat_list, dd_dataset, [dd_cat, p_ckpt], api_name=False)

        def empty_case_payload():
            return None, "", "", "", "{}"

        def sample_payload_from_choice(idx_str):
            if not idx_str or not global_sys.data_manager:
                return empty_case_payload()
            idx = int(idx_str.split("|")[0].strip())
            img, q, t, gt, opts = global_sys.data_manager.get_sample_by_idx(idx)
            return img, q, t, gt, json.dumps(opts, indent=2, ensure_ascii=False)

        # 3. Load Dataset & Reset Subclass/Logic Dropdowns
        def on_cat_load(c):
            if not global_sys.data_manager:
                return (
                    gr.update(choices=["All"], value="All"),
                    gr.update(choices=["All"], value="All"),
                    gr.update(choices=[], value=None),
                    *empty_case_payload(),
                    "Initialize the engine before loading a category.",
                )
            msg = global_sys.data_manager.load_subclass(c)
            # FORCE RESET of dropdowns to prevent cross-dataset sticky values
            subs = ["All"] + global_sys.data_manager.subfolder_choices
            logics = ["All"] + global_sys.data_manager.task_type_choices
            sample_choices = global_sys.data_manager.filter_samples()
            selected_sample = sample_choices[0] if sample_choices else None
            return (
                gr.update(choices=subs, value="All"),
                gr.update(choices=logics, value="All"),
                gr.update(choices=sample_choices, value=selected_sample),
                *sample_payload_from_choice(selected_sample),
                msg,
            )

        btn_load.click(
            on_cat_load,
            dd_cat,
            [dd_sub_type, dd_logic, dd_samples_list, img_preview_in, txt_q_in, txt_hidden_t_type, txt_gt_out, txt_opts_json, status_box],
            api_name=False,
        )

        # 4. Filter Samples
        def on_filter_update(s_f, l_f, search_t):
            if not global_sys.data_manager: return gr.update(choices=[])
            choices = global_sys.data_manager.filter_samples(
                subfolder=None if s_f == "All" else s_f,
                task_type=None if l_f == "All" else l_f,
                search_query=search_t
            )
            return gr.update(choices=choices, value=choices[0] if choices else None)

        for trigger in [dd_sub_type, dd_logic, txt_search_case]:
            trigger.change(on_filter_update, [dd_sub_type, dd_logic, txt_search_case], dd_samples_list, api_name=False)

        # 5. Select Case & Update Options State
        def on_select_case_id(idx_str):
            return sample_payload_from_choice(idx_str)

        dd_samples_list.change(
            on_select_case_id, dd_samples_list, 
            [img_preview_in, txt_q_in, txt_hidden_t_type, txt_gt_out, txt_opts_json],
            api_name=False,
        )

        # 6. Run Actions
        btn_run_main.click(
            search_runner_wrapper,
            inputs=[img_preview_in, txt_q_in, dd_cat, txt_hidden_t_type, txt_opts_json, txt_gt_out],
            outputs=[gal_rag_source, txt_rag_manual, img_hm_out, gal_sam3_preview, file_sam3_editor, txt_logs_stream, md_final_res, txt_cot_edit, state_redbox_pil, state_crops_paths, img_logic_debug, md_p1_prompt, md_method_process],
            api_name=False,
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
                txt_opts_json,
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
            api_name=False,
        )

    return demo

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch the GLLS Gradio visualization UI.")
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
    print(f"GLLS QA Method Desk launched at http://localhost:{port}")
    patch_starlette_template_response_compat()
    app = create_ui()
    app.queue(max_size=args.max_queue).launch(server_name=args.host, server_port=port, show_api=args.show_api)


if __name__ == "__main__":
    main()
