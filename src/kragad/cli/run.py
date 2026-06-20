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

from kragad import paths as kragad_paths

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

    # Pass specific device to Localizer
    localizer = LocalizerClass(args, device=device)

    results = []
    # stats_map in worker is minimal, full aggregation happens in merge_results
    stats_map = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0}))

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== Worker {rank} Init ===\nModel: {args.model_type} (vLLM={vlm_engine.use_vllm})\nDevice: {device}\n\n")

    for i, sample in enumerate(tqdm(samples, desc=f"Worker {rank}", position=rank)):
        mcts_agent = None  # 确保变量在 try 块外部定义
        mcts_output = {}   # 确保变量在 try 块外部定义
        try:
            if sample.get("annotation") is not True:
                continue
            
            task_type = sample["task_type"]
            if task_type.startswith("Object"): continue 
            image_path = sample["image_path"]
            if not os.path.exists(image_path): continue
                
            img_pil = Image.open(image_path).convert("RGB")
            gt_key = sample["gt_answer"] 
            options = sample.get("options", {})
            gt_content = options.get(gt_key, gt_key)
            current_subclass = sample["subclass"]

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
            
            # === 修改开始: 增强 Debug 信息 ===
            try:
                mcts_output = await mcts_agent.process()
            except Exception as e:
                # 1. 在控制台打印醒目的错误头，包含当前 Worker 和 样本索引
                print(f"\n🔥🔥🔥 [Worker {rank}] CRITICAL ERROR processing sample {i}!")
                print(f"Image: {sample['image_path']}")
                print(f"Question: {sample['question']}")
                
                # 2. 打印完整的堆栈信息 (这才是 debug 的关键)
                traceback.print_exc()
                
                # 3. 将详细错误写入单独的错误日志文件 (避免控制台刷屏看不清)
                error_entry = {
                    "sample_index": i,
                    "image": sample["image_path"],
                    "error_msg": str(e),
                    "traceback": traceback.format_exc()
                }
                # 使用追加模式写入 jsonl 或者简单追加到错误列表文件
                try:
                    with open(error_file, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps(error_entry) + "\n")
                except:
                    pass

                # 4. 保持原有的 fallback 逻辑，防止整个程序崩溃
                mcts_output = {
                    "text": "Error", 
                    "error": str(e), 
                    "prompt": f"SYSTEM ERROR: {str(e)}\nTraceback logged."
                }
            # === 修改结束 ===~

            final_content = []
            instruction_text = mcts_output.get("prompt", "")
            strategy = mcts_output.get("verification_strategy", getattr(args, "verification_strategy", "staged"))

            if "rag_content_list" in mcts_output:
                final_content.extend(mcts_output["rag_content_list"])

            final_content.append({"type": "text", "text": "\n=== INSPECTION TASK ===\n"})
            if "red_box_image" in mcts_output:
                final_content.append({"type": "text", "text": "--- IMAGE 1: GLOBAL VIEW ---\n"})
                final_content.append({"type": "image", "image": mcts_output["red_box_image"]})

            if "crop_images" in mcts_output:
                final_content.append({"type": "text", "text": "\n--- IMAGE 2: FOCUS VIEWS ---\n"})
                for c in mcts_output["crop_images"]:
                    final_content.append({"type": "image", "image": c})

            final_content.append({"type": "text", "text": "\n--- INSTRUCTION ---\n" + instruction_text})
            
            response = vlm_engine.generate(final_content, max_tokens=4096)

            match = re.search(r"The correct answer is \(?([A-D])\)?", response)
            if match: pred_key = match.group(1)
            else:
                cands = re.findall(r"\b([A-D])\b", response)
                pred_key = cands[-1] if cands else "A"

            is_correct = (pred_key == gt_key)
            stats_map[current_subclass][task_type]["total"] += 1
            if is_correct: stats_map[current_subclass][task_type]["correct"] += 1

            log_str = f"[{current_subclass}][{task_type}] Pred:{pred_key} GT:{gt_key} {'✅' if is_correct else '❌'}"
            with open(log_file, "a", encoding="utf-8") as f: f.write(log_str + "\n")
            
            debug_meta = mcts_output.get("debug_metadata", {})
            
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
                "trace": {
                    "verification_strategy": strategy,
                    "heatmap_score": debug_meta.get("heatmap_peak_score", 0),
                    "threshold_used": debug_meta.get("anomaly_threshold", 0),
                    "phase_1_result": debug_meta.get("phase_1_conclusion", "N/A"),
                    "did_crop": debug_meta.get("is_crop_triggered", False),
                    "crop_count": debug_meta.get("crop_count", 0),
                    "final_prompt": instruction_text,
                    "raw_response": response
                }
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
            # === [CRITICAL FIX: 显存泄漏修复核心] ===

            # 1. 手动销毁 MCTS 树（切断循环引用）
            if mcts_agent:
                try:
                    mcts_agent.cleanup()
                except Exception as e:
                    print(f"⚠️ MCTS cleanup error: {e}")

            # 2. 删除 MCTS 输出中的图像引用（这些是PIL图像，占用大量内存）
            if 'mcts_output' in locals() and isinstance(mcts_output, dict):
                # 清理红框标注图
                if 'red_box_image' in mcts_output:
                    del mcts_output['red_box_image']
                # 清理裁剪图列表
                if 'crop_images' in mcts_output:
                    for img in mcts_output.get('crop_images', []):
                        del img
                    del mcts_output['crop_images']
                # 清理RAG内容列表（可能包含多个参考图像）
                if 'rag_content_list' in mcts_output:
                    for item in mcts_output.get('rag_content_list', []):
                        if isinstance(item, dict) and 'image' in item:
                            del item['image']
                    del mcts_output['rag_content_list']
                # 清理logic debug图像
                if 'logic_debug_info' in mcts_output:
                    if isinstance(mcts_output['logic_debug_info'], dict):
                        if 'view_image' in mcts_output['logic_debug_info']:
                            del mcts_output['logic_debug_info']['view_image']
                    del mcts_output['logic_debug_info']

            # 3. 删除当前循环产生的重对象引用
            if 'mcts_agent' in locals():
                del mcts_agent
            if 'mcts_output' in locals():
                del mcts_output

            # 4. 清理 final_content（包含5-10张图像的列表）
            if 'final_content' in locals():
                for item in final_content:
                    if isinstance(item, dict) and 'image' in item:
                        del item['image']
                del final_content

            # 5. 清理其他图像引用
            if 'img_pil' in locals():
                del img_pil
            if 'response' in locals():
                del response
            if 'row' in locals() and isinstance(row, dict) and 'image' in row:
                del row['image']
                del row

            # 6. 周期性清理SAM缓存（每5个样本）
            if i % 50 == 0 and sam_engine is not None:
                try:
                    # SAM3可能有reset_image或clear_cache方法
                    if hasattr(sam_engine, 'predictor'):
                        if hasattr(sam_engine.predictor, 'reset_image'):
                            sam_engine.predictor.reset_image()
                        # 清理可能的图像缓存
                        if hasattr(sam_engine.predictor, 'features'):
                            sam_engine.predictor.features = None
                        if hasattr(sam_engine.predictor, 'original_size'):
                            sam_engine.predictor.original_size = None
                        if hasattr(sam_engine.predictor, 'input_size'):
                            sam_engine.predictor.input_size = None
                except Exception as e:
                    pass  # SAM清理失败不影响主流程

            # 7. 强制运行 Python 垃圾回收（清理 CPU 内存中的循环引用对象）
            gc.collect()

            # 8. 清理 PyTorch 显存缓存
            torch.cuda.empty_cache()

            # 9. 周期性深度同步清理（每10个样本）
            if i % 100 == 0:
                torch.cuda.synchronize()  # 确保所有CUDA操作完成
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
    from kragad.rag.agent import SimInspecAgent
    from kragad.models.mcts_sam import MCTSQuestionSample
    from kragad.models.localizer import ABounD_Localizer, AdaptCLIP_Localizer    
    from kragad.seg.sam3_engine import Sam3Engine
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

    # === [AUTO SELECTION LOGIC] ===
    if args.k_shot == 0:
        print(f"🤖 [Worker {rank}] k_shot=0 detected. Using AdaptCLIP (Zero-Shot).")
        LocalizerClass = AdaptCLIP_Localizer
    else:
        print(f"🧠 [Worker {rank}] k_shot={args.k_shot} detected. Using ABounD (Few-Shot).")
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
    
    # 统计数据结构
    detailed_stats = defaultdict(lambda: defaultdict(lambda: {
        "total": 0, "correct": 0,
        "normal_total": 0, "normal_correct": 0,
        "abnormal_total": 0, "abnormal_correct": 0
    }))
    
    # === 错题收集结构 (按类别和任务分类) ===
    # {subclass: {task_type: [list of wrong samples]}}
    wrong_samples_collector = defaultdict(lambda: defaultdict(list))
    
    wrong_samples_root = os.path.join(output_dir, "wrong_samples")
    
    print("Merging results and generating debug files...")
    
    # 1. 收集所有结果
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

    # 2. 遍历结果进行统计和错题收集
    for res in tqdm(all_results, desc="Processing Results"):
        subclass = res["subclass"]
        task_type = res["task_type"]
        is_correct = res["correct"]
        is_normal = res["is_normal"]
        
        # 基础统计
        stats_entry = detailed_stats[subclass][task_type]
        stats_entry["total"] += 1
        if is_correct:
            stats_entry["correct"] += 1
        else:
            # === 将错题加入收集器 ===
            wrong_samples_collector[subclass][task_type].append(res)
            
        # 详细统计 (用于计算平衡正确率)
        if is_normal:
            stats_entry["normal_total"] += 1
            if is_correct:
                stats_entry["normal_correct"] += 1
        else:
            stats_entry["abnormal_total"] += 1
            if is_correct:
                stats_entry["abnormal_correct"] += 1

    # === 3. 统一写入错题文件 ===
    # 遍历收集器，按 {Category}/{Task_Type}.json 格式写入
    for category, task_map in wrong_samples_collector.items():
        # 创建 Category 目录 (例如 wrong_samples/leather/)
        cat_dir = os.path.join(wrong_samples_root, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        for task_type, samples in task_map.items():
            # 写入 Task_Type.json (例如 wrong_samples/leather/anomaly_detection.json)
            # 为了防止文件名非法字符，简单处理一下
            safe_task_name = "".join([c if c.isalnum() or c in ['_','-'] else '_' for c in task_type])
            file_path = os.path.join(cat_dir, f"{safe_task_name}.json")
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=4)

    # --- 保存文件 4: 纯结果列表 (官方格式) ---
    official_json_path = os.path.join(output_dir, "final_official_format.json")
    with open(official_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    # --- 生成 FINAL REPORT TEXT ---
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
            
            # 记录错题数量到 stats summary
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

    # --- 总体统计部分 ---
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


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 动态导入加载器
    from kragad.data.dataset_loader import DSMVTecDatasetLoader, VisaDatasetLoader
    
    database_root = kragad_paths.database_root()
    graph_cache_root = args.graph_cache_root or os.path.join(database_root, "graph_index")

    # === 数据集选择逻辑 ===
    # 自动补充数据集路径
    if args.dataset.lower() == 'mvtec':
        if "DS-MVTec" not in args.dataset_root:
             args.dataset_root = os.path.join(args.dataset_root, "DS-MVTec")
        loader = DSMVTecDatasetLoader(root_path=args.dataset_root)
        
    elif args.dataset.lower() == 'visa':
        if "VisA" not in args.dataset_root:
            args.dataset_root = os.path.join(args.dataset_root, "VisA")
        loader = VisaDatasetLoader(root_path=args.dataset_root)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"🚀 Loading Dataset: {args.dataset.upper()} (Subclass: {args.subclass})")
    print(f"📂 Dataset Root: {args.dataset_root}")

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
    
    if args.limit > 0: 
        all_samples = all_samples[:args.limit]
    
    total_samples = len(all_samples)
    print(f"Total samples: {total_samples}")
    if total_samples == 0:
        print("❌ No samples found. Exiting.")
        return

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

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    # === 通用参数 ===
    parser.add_argument("--gpus", type=str, default="7")
    parser.add_argument("--model_path", default=kragad_paths.vlm_model_path())
    parser.add_argument("--model_type", type=str, default="qwen3")
    parser.add_argument("--use_vllm", action='store_true', default=True)
    parser.add_argument("--sam_path", default=kragad_paths.sam3_path())
    
    # === 数据集相关 ===
    parser.add_argument("--dataset", type=str, default='visa', choices=['mvtec', 'visa'], 
                        help="Choose dataset: mvtec or visa")
    parser.add_argument("--dataset_root", default=kragad_paths.dataset_root())
    parser.add_argument("--subclass", type=str, default="pcb4")
    parser.add_argument("--output_dir", default="./a_pcb4_qwen3/")
    parser.add_argument("--graph_cache_root", default=None,
                        help="Graph index directory. Defaults to $GLLS_DATABASE_ROOT/graph_index.")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument(
        "--verification_strategy",
        type=str,
        default="staged",
        choices=["staged", "single_prompt_concat"],
        help="Final verification strategy: staged (default) or single_prompt_concat (legacy comparison mode)."
    )
    # === Localizer 关键参数 ===
    parser.add_argument("--k_shot", type=int, default=1, help="Shot number")
    parser.add_argument("--checkpoint_path", type=str, default=kragad_paths.abound_model_path())
    parser.add_argument("--save_path", type=str, default=kragad_paths.abound_save_path())
    parser.add_argument("--image_size", type=int, default=336)
    
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    
    main(args)
