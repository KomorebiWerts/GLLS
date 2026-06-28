import os
import gc
import argparse
import json
import asyncio
import re
import math
import sys
import traceback
import random
import numpy as np
import base64
from io import BytesIO
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import torch.multiprocessing as mp

from glls import paths as glls_paths
from glls.runtime_config import resolve_localizer_name, runtime_weight_config
from glls.qa_trace import build_method_trace, binary_anomaly_decision_override

PAPER_SCOPE_TASK_TYPES = {
    "Anomaly Detection",
    "Defect Classification",
    "Defect Localization",
    "Defect Description",
    "Defect Analysis",
}

# ==========================================
# 0. Environment & Setup
# ==========================================
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_CONFIGURE_LOGGING"] = "0" 

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # torch/cuda seeds set inside worker

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def is_paper_scope_task(task_type):
    return str(task_type or "").strip() in PAPER_SCOPE_TASK_TYPES


def parse_task_type_filter(task_types):
    if not task_types:
        return None
    selected = {
        item.strip()
        for item in str(task_types).split(",")
        if item.strip()
    }
    unknown = selected - PAPER_SCOPE_TASK_TYPES
    if unknown:
        valid = ", ".join(sorted(PAPER_SCOPE_TASK_TYPES))
        raise ValueError(f"Unknown --task_types value(s): {sorted(unknown)}. Valid values: {valid}")
    return selected


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_normal_support_images(dataset_root, dataset_name, category, limit):
    if int(limit or 0) <= 0:
        return []

    root = os.path.abspath(str(dataset_root))
    category = str(category)
    dataset_key = str(dataset_name or "").strip().lower()
    if dataset_key == "mvtec":
        candidate_dirs = [
            os.path.join(root, category, "image", "good"),
            os.path.join(root, category, "train", "good"),
            os.path.join(root, category, "test", "good"),
        ]
    else:
        candidate_dirs = [
            os.path.join(root, category, "train", "good"),
            os.path.join(root, category, "test", "good"),
            os.path.join(root, category, "image", "good"),
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


def configure_localizer_for_category(localizer, args, category, support_cache):
    configure = getattr(localizer, "configure_support", None)
    if callable(configure):
        if category not in support_cache:
            support_cache[category] = find_normal_support_images(
                args.dataset_root,
                args.dataset,
                category,
                getattr(args, "k_shot", 0),
            )
        configure(category, support_cache[category])
        return

    set_category = getattr(localizer, "set_active_category", None)
    if callable(set_category):
        set_category(category)


def resolve_localizer_choice(args):
    return resolve_localizer_name(
        getattr(args, "localizer", "auto"),
        getattr(args, "dataset", ""),
        getattr(args, "k_shot", 0),
    )


def configure_localizer_args(args, localizer_name):
    adaptclip_ckpt_path = getattr(args, "checkpoint_path", "")
    if localizer_name == "adaptclip" and not adaptclip_ckpt_path:
        domain = str(getattr(args, "adaptclip_checkpoint_domain", "auto") or "auto").strip().lower()
        if domain != "auto":
            adaptclip_ckpt_path = os.path.join(
                glls_paths.adaptclip_root(),
                "checkpoints",
                f"{domain}_epoch_15.pth",
            )
    cfg = runtime_weight_config(
        getattr(args, "dataset", ""),
        localizer_name,
        getattr(args, "k_shot", 0),
        adaptclip_ckpt_path=adaptclip_ckpt_path,
        adaptclip_root=glls_paths.adaptclip_root(),
        abound_model_path=getattr(args, "checkpoint_path", "") or glls_paths.abound_model_path(),
        abound_save_path=getattr(args, "save_path", "") or glls_paths.abound_save_path(),
    )
    args.checkpoint_path = cfg["checkpoint_path"]
    args.save_path = cfg["save_path"]
    if int(getattr(args, "image_size", 0) or 0) <= 0:
        args.image_size = int(cfg["image_size"])


# ==========================================
# 1. Unified Inference Class (Lazy Imports)
# ==========================================
class UnifiedVLMInference:
    def __init__(self, model_path, device, model_type="qwen2.5-vl", use_vllm=False,max_tokens=8192):
        self.device = device 
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.max_tokens = max_tokens
        # Check vLLM availability dynamically
        try:
            import vllm
            self.has_vllm = True
        except ImportError:
            self.has_vllm = False
            
        self.use_vllm = use_vllm and self.has_vllm

        if self.use_vllm:
            print(f"🔧 [VLM] Initializing vLLM on {device}...")
            from vllm import LLM, SamplingParams
            
            self.model = LLM(
                model=model_path,
                trust_remote_code=True,
                gpu_memory_utilization=0.60, 
                tensor_parallel_size=1,
                dtype="bfloat16",       
                max_model_len=self.max_tokens,     
                enforce_eager=False,
                disable_custom_all_reduce=True
            )
            self.sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
            return

        # --- Transformers Fallback (Lazy Import) ---
        import torch 
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        if "qwen3" in self.model_type:
            try:
                from transformers import Qwen3VLForConditionalGeneration
                model_cls = Qwen3VLForConditionalGeneration
            except ImportError:
                model_cls = AutoModelForCausalLM
            
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = model_cls.from_pretrained(
                model_path, torch_dtype="auto", device_map=device, trust_remote_code=True
            )
        elif "qwen" in self.model_type: 
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
            )
        elif "llava" in self.model_type:
            from transformers import LlavaOnevisionForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=device
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.eval()

    def generate(self, content_list, max_tokens=4096):
        import torch
        from PIL import Image
        
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
        import torch
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = [item["image"] for item in content if item["type"] == "image"]
        
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
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
        import torch
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
        return decoded.split("assistant\n")[-1] if "assistant\n" in decoded else decoded

# ==========================================
# 2. Worker Async Logic
# ==========================================
async def run_worker_async(rank, device, samples, args, graph_cache_root, log_file, error_file, temp_dir, sam_engine, LocalizerClass, RagAgentClass, MCTSClass):
    import torch 
    from PIL import Image
    
    # Initialize VLM
    vlm_engine = UnifiedVLMInference(
        model_path=args.model_path, 
        device=device,
        model_type=args.model_type,
        use_vllm=args.use_vllm,
        max_tokens=args.max_tokens
    )

    rag_agent = RagAgentClass(vlm_engine=None, cache_root=graph_cache_root, k_shot=args.k_shot)
    rag_cache = {}

    configure_localizer_args(args, getattr(args, "resolved_localizer", resolve_localizer_choice(args)))

    # Pass specific device to Localizer
    if getattr(args, "resolved_localizer", "") == "adaptclip":
        localizer = LocalizerClass(args, device=device, pretrained_model=args.pretrained_model)
    else:
        localizer = LocalizerClass(args, device=device)
    support_cache = {}

    results = []
    # stats_map in worker is minimal, full aggregation happens in merge_results
    stats_map = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0}))

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== Worker {rank} Init ===\nModel: {args.model_type} (vLLM={vlm_engine.use_vllm})\nDevice: {device}\n\n")

    for i, sample in enumerate(tqdm(samples, desc=f"Worker {rank}", position=rank)):
        mcts_agent = None
        mcts_output = {}
        try:
            if sample.get("annotation") is not True:
                continue
            
            task_type = sample["task_type"]
            if not is_paper_scope_task(task_type):
                continue
            image_path = sample["image_path"]
            if not os.path.exists(image_path): continue
                
            img_pil = Image.open(image_path).convert("RGB")
            gt_key = sample["gt_answer"] 
            options = sample.get("options", {})
            gt_content = options.get(gt_key, gt_key)
            current_subclass = sample["subclass"]
            configure_localizer_for_category(localizer, args, current_subclass, support_cache)

            row = {
                "index": i, "image": img_pil, "question": sample["question"],
                "answer": gt_content, "options": options,
                "category": current_subclass, "type": task_type
            }

            mcts_agent = MCTSClass(
                row=row, 
                args=args, 
                inference_engine=vlm_engine, 
                localizer=localizer, 
                rag_agent=rag_agent,  
                rag_cache=rag_cache,  
                rag_blocks=None,
                sam_engine=sam_engine
            )
            
            try:
                mcts_output = await mcts_agent.process()
            except Exception as e:
                print(f"\n🔥🔥🔥 [Worker {rank}] CRITICAL ERROR processing sample {i}!")
                print(f"Image: {sample['image_path']}")
                print(f"Question: {sample['question']}")
                
                traceback.print_exc()
                
                error_entry = {
                    "sample_index": i,
                    "image": sample["image_path"],
                    "error_msg": str(e),
                    "traceback": traceback.format_exc()
                }
                try:
                    with open(error_file, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps(error_entry) + "\n")
                except:
                    pass

                mcts_output = {
                    "text": "Error", 
                    "error": str(e), 
                    "prompt": f"SYSTEM ERROR: {str(e)}\nTraceback logged."
                }

            final_content = []
            instruction_text = mcts_output.get("prompt", "")
            strategy = mcts_output.get("verification_strategy", "staged")

            if "rag_content_list" in mcts_output:
                final_content.extend(mcts_output["rag_content_list"])

            final_content.append({"type": "text", "text": "\n=== INSPECTION TASK ===\n"})
            if "red_box_image" in mcts_output:
                final_content.append({"type": "text", "text": "--- IMAGE 1: GLOBAL VIEW ---\n"})
                final_content.append({"type": "image", "image": mcts_output["red_box_image"]})

            show_crops_in_prompt = bool(mcts_output.get("show_crop_images_in_final_prompt", True))
            prompt_crop_images = mcts_output.get("prompt_crop_images", mcts_output.get("crop_images", []))
            if show_crops_in_prompt and prompt_crop_images:
                final_content.append({"type": "text", "text": "\n--- IMAGE 2: FOCUS VIEWS ---\n"})
                crop_labels = mcts_output.get("prompt_crop_labels", mcts_output.get("crop_labels", []))
                suppress_crop_labels = bool(mcts_output.get("suppress_prompt_crop_labels", False))
                for crop_idx, c in enumerate(prompt_crop_images, start=1):
                    if not suppress_crop_labels:
                        label = ""
                        if crop_idx - 1 < len(crop_labels):
                            label = str(crop_labels[crop_idx - 1]).strip()
                        if not label:
                            label = f"Focus View {crop_idx}"
                        final_content.append({"type": "text", "text": f"Focus View {crop_idx}: {label}\n"})
                    final_content.append({"type": "image", "image": c})

            final_content.append({"type": "text", "text": "\n--- INSTRUCTION ---\n" + instruction_text})
            
            response = vlm_engine.generate(final_content, max_tokens=4096)

            match = re.search(r"The correct answer is \(?([A-D])\)?", response)
            if match: pred_key = match.group(1)
            else:
                cands = re.findall(r"\b([A-D])\b", response)
                pred_key = cands[-1] if cands else "A"

            debug_meta = mcts_output.get("debug_metadata", {})
            ad_override = binary_anomaly_decision_override(task_type, options, debug_meta, pred_key)
            if ad_override:
                pred_key = ad_override["override_pred_key"]
                debug_meta["binary_anomaly_decision_override"] = ad_override

            is_correct = (pred_key == gt_key)
            stats_map[current_subclass][task_type]["total"] += 1
            if is_correct: stats_map[current_subclass][task_type]["correct"] += 1

            log_str = f"[{current_subclass}][{task_type}] Pred:{pred_key} GT:{gt_key} {'✅' if is_correct else '❌'}"
            with open(log_file, "a", encoding="utf-8") as f: f.write(log_str + "\n")
            
            try:
                rel_path = os.path.relpath(image_path, args.dataset_root)
            except ValueError:
                rel_path = image_path 
            
            # Determine is_normal based on path convention
            is_normal_sample = "good" in image_path.split(os.sep) or "good" in os.path.basename(os.path.dirname(image_path))

            res_entry = {
                "worker_rank": rank,
                "sample_id": i,
                "image": rel_path,
                "abs_image_path": image_path,
                "subclass": current_subclass,
                "task_type": task_type,
                "question": sample["question"],
                "correct_answer": gt_key,
                "gpt_answer": pred_key,
                "correct": is_correct,
                "is_normal": is_normal_sample, 
                "trace": build_method_trace(
                    debug_meta,
                    mcts_output,
                    strategy,
                    instruction_text,
                    response,
                    trace_level=args.trace_level,
                )
            }
            results.append(res_entry)
            
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ [Worker {rank}] Error processing sample {i}: {e}")
            traceback.print_exc()
            if "EngineDeadError" in str(e):
                break
            continue

        finally:
            # Release MCTS state and image-heavy outputs between samples.
            if mcts_agent:
                try:
                    mcts_agent.cleanup()
                except Exception as e:
                    print(f"⚠️ MCTS cleanup error: {e}")

            if 'mcts_output' in locals() and isinstance(mcts_output, dict):
                if 'red_box_image' in mcts_output:
                    del mcts_output['red_box_image']
                if 'crop_images' in mcts_output:
                    for img in mcts_output.get('crop_images', []):
                        del img
                    del mcts_output['crop_images']
                if 'prompt_crop_images' in mcts_output:
                    for img in mcts_output.get('prompt_crop_images', []):
                        del img
                    del mcts_output['prompt_crop_images']
                if 'rag_content_list' in mcts_output:
                    for item in mcts_output.get('rag_content_list', []):
                        if isinstance(item, dict) and 'image' in item:
                            del item['image']
                    del mcts_output['rag_content_list']
                if 'logic_debug_info' in mcts_output:
                    if isinstance(mcts_output['logic_debug_info'], dict):
                        if 'view_image' in mcts_output['logic_debug_info']:
                            del mcts_output['logic_debug_info']['view_image']
                    del mcts_output['logic_debug_info']

            if 'mcts_agent' in locals():
                del mcts_agent
            if 'mcts_output' in locals():
                del mcts_output

            if 'final_content' in locals():
                for item in final_content:
                    if isinstance(item, dict) and 'image' in item:
                        del item['image']
                del final_content

            if 'img_pil' in locals():
                del img_pil
            if 'response' in locals():
                del response
            if 'row' in locals() and isinstance(row, dict) and 'image' in row:
                del row['image']
                del row

            if i % 50 == 0 and sam_engine is not None:
                try:
                    if hasattr(sam_engine, 'predictor'):
                        if hasattr(sam_engine.predictor, 'reset_image'):
                            sam_engine.predictor.reset_image()
                        if hasattr(sam_engine.predictor, 'features'):
                            sam_engine.predictor.features = None
                        if hasattr(sam_engine.predictor, 'original_size'):
                            sam_engine.predictor.original_size = None
                        if hasattr(sam_engine.predictor, 'input_size'):
                            sam_engine.predictor.input_size = None
                except Exception as e:
                    pass

            gc.collect()

            torch.cuda.empty_cache()

            if i % 100 == 0:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

    final_output_path = os.path.join(temp_dir, f"results_rank_{rank}.json")
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats_map, "results": results}, f, indent=2)

# ==========================================
# 3. Worker Entry
# ==========================================
def worker_entry(rank, gpu_ids, all_chunks, args, graph_cache_root):
    # --- CRITICAL: Set Isolation BEFORE any torch/cuda imports ---
    target_gpu_id = gpu_ids[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id)
    
    # Debug info
    print(f"🚀 [Worker {rank}] Setting CUDA_VISIBLE_DEVICES={target_gpu_id}")

    # --- NOW we can import torch and other heavy libraries ---
    import torch
    from glls.rag.agent import SimInspecAgent
    from glls.models.mcts_sam import MCTSQuestionSample
    from glls.models.localizer import ABounD_Localizer, AdaptCLIP_Localizer    
    from glls.seg.sam3_engine import Sam3Engine
    # ---------------------------------------------------------

    # Re-seed after torch import
    setup_seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # After isolation, the visible device is always 'cuda:0'
    device_str = "cuda:0"
    
    my_samples = all_chunks[rank]
    temp_dir = os.path.join(args.output_dir, "temp_results")
    log_file = os.path.join(args.output_dir, f"worker_{rank}.log")
    error_file = os.path.join(args.output_dir, f"worker_{rank}_errors.json")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"🚀 [Worker {rank}] Running on Logic Device: {device_str} (Physical: {target_gpu_id}) | Samples: {len(my_samples)}")

    # Load SAM3
    sam_engine = None
    if args.sam_path and os.path.exists(args.sam_path):
        try:
            print(f"🔌 [Worker {rank}] Loading SAM3 to {device_str}...")
            sam_engine = Sam3Engine(args.sam_path, device=device_str)
        except Exception as e:
            print(f"⚠️ [Worker {rank}] Failed to load SAM3: {e}")

    # === [LOCALIZER SELECTION LOGIC] ===
    localizer_name = resolve_localizer_choice(args)
    args.resolved_localizer = localizer_name
    configure_localizer_args(args, localizer_name)
    if localizer_name == "adaptclip":
        mode = "Few-Shot" if int(args.k_shot or 0) > 0 else "Zero-Shot"
        print(f"🤖 [Worker {rank}] Using AdaptCLIP ({mode}, k_shot={args.k_shot}).")
        LocalizerClass = AdaptCLIP_Localizer
    else:
        print(f"🧠 [Worker {rank}] Using ABounD (k_shot={args.k_shot}).")
        LocalizerClass = ABounD_Localizer

    try:
        asyncio.run(run_worker_async(
            rank=rank, 
            device=device_str, 
            samples=my_samples, 
            args=args, 
            graph_cache_root=graph_cache_root,
            log_file=log_file,
            error_file=error_file,
            temp_dir=temp_dir,
            sam_engine=sam_engine,
            LocalizerClass=LocalizerClass, # <--- Passed dynamically
            RagAgentClass=SimInspecAgent,
            MCTSClass=MCTSQuestionSample
        ))
    except Exception as e:
        print(f"🔥 [Worker {rank}] CRITICAL CRASH: {e}")
        traceback.print_exc()

def merge_results(output_dir, num_workers):
    temp_dir = os.path.join(output_dir, "temp_results")
    all_results = []
    
    detailed_stats = defaultdict(lambda: defaultdict(lambda: {
        "total": 0, "correct": 0,
        "normal_total": 0, "normal_correct": 0,
        "abnormal_total": 0, "abnormal_correct": 0
    }))
    
    # {subclass: {task_type: [list of wrong samples]}}
    wrong_samples_collector = defaultdict(lambda: defaultdict(list))
    
    wrong_samples_root = os.path.join(output_dir, "wrong_samples")
    
    print("Merging results and generating debug files...")
    
    for i in range(num_workers):
        res_file = os.path.join(temp_dir, f"results_rank_{i}.json")
        if os.path.exists(res_file):
            try:
                with open(res_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "results" in data:
                        all_results.extend(data["results"])
            except Exception as e:
                print(f"Error reading {res_file}: {e}")

    for res in tqdm(all_results, desc="Processing Results"):
        subclass = res["subclass"]
        task_type = res["task_type"]
        is_correct = res["correct"]
        is_normal = res["is_normal"]
        
        stats_entry = detailed_stats[subclass][task_type]
        stats_entry["total"] += 1
        if is_correct:
            stats_entry["correct"] += 1
        else:
            wrong_samples_collector[subclass][task_type].append(res)
            
        if is_normal:
            stats_entry["normal_total"] += 1
            if is_correct:
                stats_entry["normal_correct"] += 1
        else:
            stats_entry["abnormal_total"] += 1
            if is_correct:
                stats_entry["abnormal_correct"] += 1

    for category, task_map in wrong_samples_collector.items():
        cat_dir = os.path.join(wrong_samples_root, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        for task_type, samples in task_map.items():
            safe_task_name = "".join([c if c.isalnum() or c in ['_','-'] else '_' for c in task_type])
            file_path = os.path.join(cat_dir, f"{safe_task_name}.json")
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=4)

    official_json_path = os.path.join(output_dir, "final_official_format.json")
    with open(official_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    final_stats_summary = {}
    report_path = os.path.join(output_dir, "final_report.txt")
    report_content = "============================================================\n"
    report_content += f"FINAL ACCURACY REPORT | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_content += "============================================================\n\n"
    
    global_metrics = defaultdict(lambda: {
        "correct": 0, "total": 0, 
        "bal_acc_sum": 0.0, "bal_acc_count": 0
    })
    total_correct_all = 0
    total_count_all = 0

    sorted_categories = sorted(detailed_stats.keys())
    
    for category in sorted_categories:
        tasks = detailed_stats[category]
        report_content += f"Category: {category}\n"
        report_content += "-" * 90 + "\n"
        report_content += f"{'Task Type':<30} | {'Acc':<8} | {'Count':<10} | {'Bal. Acc':<10} | {'Norm/Abn Acc'}\n"
        report_content += "-" * 90 + "\n"
        
        cat_correct = 0
        cat_total = 0
        final_stats_summary[category] = {}

        for task_type in sorted(tasks.keys()):
            s = tasks[task_type]
            
            acc = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0.0
            
            norm_acc = (s["normal_correct"] / s["normal_total"]) if s["normal_total"] > 0 else 0.0
            abn_acc = (s["abnormal_correct"] / s["abnormal_total"]) if s["abnormal_total"] > 0 else 0.0
            
            # Balanced Accuracy Calculation
            if s["normal_total"] > 0 and s["abnormal_total"] > 0:
                balanced_acc = (norm_acc + abn_acc) / 2.0
            elif s["normal_total"] > 0:
                balanced_acc = norm_acc
            else:
                balanced_acc = abn_acc
            bal_acc_percent = balanced_acc * 100
            
            details_str = f"N:{norm_acc*100:.1f}% A:{abn_acc*100:.1f}%"
            report_content += f"{task_type:<30} | {acc:.1f}%    | {s['correct']}/{s['total']:<5} | {bal_acc_percent:.1f}%     | {details_str}\n"
            
            cat_correct += s["correct"]
            cat_total += s["total"]
            
            global_metrics[task_type]["correct"] += s["correct"]
            global_metrics[task_type]["total"] += s["total"]
            global_metrics[task_type]["bal_acc_sum"] += bal_acc_percent
            global_metrics[task_type]["bal_acc_count"] += 1
            
            wrong_count = len(wrong_samples_collector[category][task_type])
            
            final_stats_summary[category][task_type] = {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": round(acc, 2),
                "balanced_accuracy": round(bal_acc_percent, 2),
                "normal_accuracy": round(norm_acc * 100, 2),
                "abnormal_accuracy": round(abn_acc * 100, 2),
                "wrong_count": wrong_count
            }

        cat_avg = (cat_correct / cat_total * 100) if cat_total > 0 else 0.0
        report_content += "-" * 90 + "\n"
        report_content += f"{category} Overall Average: {cat_avg:.1f}% ({cat_correct}/{cat_total})\n\n"
        
        total_correct_all += cat_correct
        total_count_all += cat_total

    report_content += "============================================================\n"
    overall_avg = (total_correct_all / total_count_all * 100) if total_count_all > 0 else 0.0
    report_content += f"OVERALL AVERAGE: {overall_avg:.2f}% ({total_correct_all}/{total_count_all})\n"
    report_content += "============================================================\n"
    report_content += f"{'Task Type':<30} | {'Acc':<8} | {'Count':<10} | {'Avg Bal. Acc'}\n"
    report_content += "-" * 90 + "\n"
    
    for task_type in sorted(global_metrics.keys()):
        m = global_metrics[task_type]
        acc = (m["correct"] / m["total"] * 100) if m["total"] > 0 else 0.0
        avg_bal_acc = (m["bal_acc_sum"] / m["bal_acc_count"]) if m["bal_acc_count"] > 0 else 0.0
        report_content += f"{task_type:<30} | {acc:.1f}%    | {m['correct']}/{m['total']:<5} | {avg_bal_acc:.1f}%\n"
    report_content += "-" * 90 + "\n"

    with open(stats_json_path := os.path.join(output_dir, "final_stats_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_stats_summary, f, indent=4)
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"✅ Merged {len(all_results)} samples.")
    print(f"📂 Wrong samples saved to folder: {wrong_samples_root}")
    print(f"📄 Results saved to: {official_json_path}")
    print(f"📊 Statistics saved to: {stats_json_path}")
    print(f"📝 Final Report saved to: {report_path}")


def _resolve_qa_root_for_dataset(qa_root, dataset_folder):
    if not qa_root:
        return None
    raw_qa_root = os.path.normpath(qa_root)
    if not os.path.exists(raw_qa_root):
        return None
    if os.path.basename(raw_qa_root) == dataset_folder:
        final_qa_root = raw_qa_root
    else:
        final_qa_root = os.path.join(raw_qa_root, dataset_folder)
    return final_qa_root if os.path.exists(final_qa_root) else None


def _looks_like_local_path(path: str) -> bool:
    text = str(path or "")
    return text.startswith(("/", "~", "."))


def validate_model_path_for_run(model_path: str) -> None:
    if _looks_like_local_path(model_path) and not os.path.exists(os.path.expanduser(model_path)):
        raise FileNotFoundError(
            f"Local VLM model path does not exist: {model_path}. "
            "Pass --model_path or set GLLS_VLM_MODEL_PATH."
        )


def assert_complete_result_count(output_dir: str, expected_total: int) -> None:
    result_path = os.path.join(output_dir, "final_official_format.json")
    if not os.path.exists(result_path):
        raise RuntimeError(f"Missing final result file after run: {result_path}")
    with open(result_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    actual_total = len(rows) if isinstance(rows, list) else 0
    if actual_total != expected_total:
        raise RuntimeError(
            f"Incomplete run: expected {expected_total} result(s), got {actual_total}. "
            "Check worker logs and *_errors.json before using this output."
        )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    from glls.data.dataset_loader import DSMVTecDatasetLoader, VisaDatasetLoader
    
    database_root = glls_paths.database_root()
    graph_cache_root = args.graph_cache_root or os.path.join(database_root, "graph_index")

    if args.dataset.lower() == 'mvtec':
        if "DS-MVTec" not in args.dataset_root:
             args.dataset_root = os.path.join(args.dataset_root, "DS-MVTec")
        qa_root = _resolve_qa_root_for_dataset(getattr(args, "qa_root", None), "DS-MVTec")
        loader = DSMVTecDatasetLoader(root_path=args.dataset_root, qa_root_path=qa_root)
        
    elif args.dataset.lower() == 'visa':
        if "VisA" not in args.dataset_root:
            args.dataset_root = os.path.join(args.dataset_root, "VisA")
        qa_root = _resolve_qa_root_for_dataset(getattr(args, "qa_root", None), "VisA")
        loader = VisaDatasetLoader(root_path=args.dataset_root, qa_root_path=qa_root)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"🚀 Loading Dataset: {args.dataset.upper()} (Subclass: {args.subclass})")
    print(f"📂 Dataset Root: {args.dataset_root}")
    print(f"📄 QA Root: {qa_root or args.dataset_root}")

    all_samples = []
    available_classes = loader.get_subclasses()
    
    if args.subclass.lower() == "all":
        target_classes = available_classes
    else:
        target_classes = [c.strip() for c in args.subclass.split(",")]

    for cls_name in target_classes:
        if cls_name not in available_classes:
            print(f"⚠️ Warning: Class '{cls_name}' not found in {args.dataset}")
            continue
        all_samples.extend(list(loader.parse_samples(cls_name)))
    
    # Keep the official evaluation path aligned with the task families reported
    # in the paper.
    task_type_filter = parse_task_type_filter(getattr(args, "task_types", None))
    all_samples = [
        sample for sample in all_samples
        if is_paper_scope_task(sample.get("task_type", ""))
        and (task_type_filter is None or sample.get("task_type") in task_type_filter)
    ]

    if args.limit > 0: 
        all_samples = all_samples[:args.limit]
    
    total_samples = len(all_samples)
    print(f"Total samples: {total_samples}")
    if total_samples == 0:
        print("❌ No samples found. Exiting.")
        return
    validate_model_path_for_run(getattr(args, "model_path", ""))

    gpu_ids = [int(x) for x in args.gpus.split(",")] if args.gpus else [0]
    num_gpus = len(gpu_ids)
    if num_gpus > 8:
        raise ValueError(
            "This launcher allows up to 8 GPUs by default. "
            "Split the run or update the guard before launching a larger job."
        )
    
    chunk_size = math.ceil(total_samples / num_gpus)
    chunks = [all_samples[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]
    while len(chunks) < num_gpus: chunks.append([])

    mp.spawn(
        worker_entry,
        args=(gpu_ids, chunks, args, graph_cache_root),
        nprocs=num_gpus,
        join=True
    )
    merge_results(args.output_dir, num_gpus)
    assert_complete_result_count(args.output_dir, total_samples)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="7")
    parser.add_argument("--model_path", default=glls_paths.vlm_model_path())
    parser.add_argument("--model_type", type=str, default="qwen3")
    parser.add_argument("--use_vllm", action='store_true', default=True)
    parser.add_argument("--sam_path", default=glls_paths.sam3_path())
    
    parser.add_argument("--dataset", type=str, default='visa', choices=['mvtec', 'visa'], 
                        help="Choose dataset: mvtec or visa")
    parser.add_argument("--dataset_root", default=glls_paths.dataset_root())
    parser.add_argument("--qa_root", default=glls_paths.qa_root())
    parser.add_argument("--subclass", type=str, default="pcb4")
    parser.add_argument("--output_dir", default="./a_pcb4_qwen3/")
    parser.add_argument("--graph_cache_root", default=None,
                        help="Graph index directory. Defaults to $GLLS_DATABASE_ROOT/graph_index.")
    parser.add_argument(
        "--task_types",
        type=str,
        default="",
        help=(
            "Optional comma-separated paper-scope QA task filter, e.g. "
            "'Defect Localization,Defect Classification'."
        ),
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument(
        "--trace_level",
        choices=["method", "full"],
        default="method",
        help="Store compact method provenance by default; use full to also save prompts/raw responses.",
    )
    parser.add_argument(
        "--localizer",
        choices=["auto", "abound", "adaptclip"],
        default="auto",
        help="auto uses the published route: MVTec/VisA 1-shot ABounD, otherwise AdaptCLIP.",
    )
    parser.add_argument("--k_shot", type=int, default=1, help="Shot number")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="AdaptCLIP checkpoint file. Defaults to $GLLS_ADAPTCLIP_ROOT/checkpoints/<domain>_epoch_15.pth.",
    )
    parser.add_argument(
        "--adaptclip_checkpoint_domain",
        choices=["auto", "mvtec", "visa"],
        default="auto",
        help="AdaptCLIP checkpoint domain used when --localizer adaptclip and --checkpoint_path is empty.",
    )
    parser.add_argument("--pretrained_model", type=str, default="ViT-L/14@336px")
    parser.add_argument("--save_path", type=str, default="", help="Local ABounD-only save path; unused by AdaptCLIP.")
    parser.add_argument("--image_size", type=int, default=0)
    
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    
    main(args)
