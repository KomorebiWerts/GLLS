import os
import re
import json
import argparse
import torch
import math
import sys
import glob
from tqdm import tqdm
from PIL import Image
import torch.multiprocessing as mp
from datetime import datetime
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoModelForVision2Seq, 
    AutoProcessor
)

# --- Path Setup ---
from kragad import paths as kragad_paths
from kragad.data.dataset_loader import DSMVTecDatasetLoader
from kragad.rag.GraphRag import SimInspecGraphEngine 
from kragad.models.localizer import ABounD_Localizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================= 0. Model Factory =======================
class ModelFactory:
    @staticmethod
    def load(model_path, model_type="auto", device_map="auto", **kwargs):
        print(f"🚀 [ModelFactory] Loading model from: {model_path}")
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️ Processor load warning: {e}")
            processor = None

        if model_type.lower() == "qwen2.5":
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True, **kwargs
            ), processor
        else:
            return AutoModelForVision2Seq.from_pretrained(
                model_path, torch_dtype="auto", device_map=device_map, trust_remote_code=True, **kwargs
            ), processor

# ======================= 1. Model Wrapper =======================
class LocalVLMInference:
    def __init__(self, model_path, device, model_type="auto"):
        self.model, self.processor = ModelFactory.load(
            model_path, 
            model_type=model_type, 
            device_map={"": device} 
        )
        self.device = device

    def generate(self, content_list, max_tokens=1024):
        # Qwen2.5-VL format
        messages = [{"role": "user", "content": content_list}]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = [item['image'] for item in content_list if item['type'] == 'image']
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0]

# ======================= 2. Agentic Logic =======================
class SimInspecAgent:
    def __init__(self, vlm_engine, cache_root, k_shot=1):
        self.vlm = vlm_engine
        self.cache_root = cache_root
        self.rag_engines = {} 
        self.k_shot = k_shot 
# 在 rag_test.py 的 SimInspecAgent 类中添加/修改以下方法

    def get_rag_context(self, subclass):
        """
        [修正后] 全量解析 RAG 数据。
        通用规则：任何值为 'N/A' 的字段都不会被包含在 Prompt 中。
        """
        rag = self._get_rag_engine(subclass)
        if not rag.G.number_of_nodes(): return []

        all_regions = [n for n, d in rag.G.nodes(data=True) if d.get("type") == "region"]
        
        rag_blocks = [] 

        for region in all_regions:
            # 1. 获取该区域的完整 info 字典
            info = rag.get_inspection_checklist(region)
            
            lines = []
            lines.append(f"--- Reference Knowledge for Region: '{region}' ---")
            
            # --- 通用过滤函数：检查内容是否有效且不是 N/A ---
            def is_valid_content(text):
                if not text: return False
                if not isinstance(text, str): return True # 非字符串默认保留
                return text.strip().upper() != "N/A"

            # 2. 基础字段 (自动过滤 N/A)
            if is_valid_content(info.get("definition")):
                lines.append(f"Definition: {info['definition']}")
            
            if is_valid_content(info.get("normal_standard")):
                lines.append(f"Normal Standard: {info['normal_standard']}")
            
            if is_valid_content(info.get("critical_check")):
                lines.append(f"Critical Check: {info['critical_check']}")
            
            # 3. 缺陷详情 (全量解析)
            raw_defects = info.get("defects", [])
            if raw_defects:
                lines.append("\n[Potential Defects Details]:")
                for d in raw_defects:
                    # 如果是字典（新版 JSON 格式），解析所有字段
                    if isinstance(d, dict):
                        d_name = d.get("type", d.get("name", "Unknown Defect"))
                        lines.append(f"\n>>> Defect: {d_name}")
                        
                        # 同样应用 N/A 过滤规则
                        if is_valid_content(d.get("visual_signature")):
                            lines.append(f"    Visual Signature: {d['visual_signature']}")
                        
                        if is_valid_content(d.get("visual_appearance")): # 兼容旧key
                            lines.append(f"    Visual Appearance: {d['visual_appearance']}")
                            
                        if is_valid_content(d.get("contrast_vs_normal")):
                            lines.append(f"    Contrast vs Normal: {d['contrast_vs_normal']}")
                            
                        # 区分项 (Distinctions)
                        if "distinctions" in d and isinstance(d["distinctions"], list):
                            for dist in d["distinctions"]:
                                if isinstance(dist, dict):
                                    diff = dist.get('difference', 'N/A')
                                    if is_valid_content(diff):
                                        lines.append(f"    * Distinction vs {dist.get('target_defect')}: {diff}")
                    
                    # 兼容旧版字符串格式
                    else:
                        lines.append(f"- {str(d)}")

            # 4. 防幻觉规则
            if "anti_hallucination_rules" in info:
                lines.append("\n[Anti-Hallucination Rules]:")
                for rule in info["anti_hallucination_rules"]:
                    if is_valid_content(rule):
                        lines.append(f"!!! {rule}")

            region_text = "\n".join(lines)

            # 5. 准备图片路径 (获取前 4 张)
            img_paths_list = []
            img_paths = rag.get_image_paths(region)
            if img_paths:
                selected_paths = img_paths[:self.k_shot] 
                for p in selected_paths:
                    if os.path.exists(p):
                        img_paths_list.append(p)
            
            # 6. 打包存入
            rag_blocks.append({
                "region": region,
                "text": region_text,
                "images": img_paths_list
            })
        
        return rag_blocks

    def _get_rag_engine(self, subclass):
        if subclass in self.rag_engines:
            return self.rag_engines[subclass]
        pkl_path = os.path.join(self.cache_root, f"{subclass}_graph.pkl")
        engine = SimInspecGraphEngine()
        if os.path.exists(pkl_path):
            engine.load_from_disk(pkl_path)
        self.rag_engines[subclass] = engine
        return engine
    
    def solve(self, sample):
        image_path = sample["image_path"]
        subclass = sample["subclass"]
        question = sample["question"]
        options = sample["options"]
        
        rag_blocks = self.get_rag_context(subclass)
        
        content_list = []
        content_list.append({"type": "text", "text": f"You are an expert QA Inspector for {subclass}.\nVerify the Test Image against the Standards below.\n"})

        if rag_blocks:
            content_list.append({"type": "text", "text": "=== 📚 REFERENCE STANDARDS (FULL DETAILS) ===\n"})
            for b in rag_blocks:
                # 1. 先放图片
                if b.get('images'):
                    content_list.append({"type": "text", "text": f"\n[Visual Reference for {b['region']}]:\n"})
                    for p in b['images']:
                        try:
                            if os.path.exists(p):
                                ref_img = Image.open(p).convert("RGB")
                                content_list.append({"type": "image", "image": ref_img})
                        except: pass
                
                # 2. 后放详细文本
                content_list.append({"type": "text", "text": f"\n{b['text']}\n"})
            content_list.append({"type": "text", "text": "=============================================\n"})

        try:
            if os.path.exists(image_path):
                test_img = Image.open(image_path).convert("RGB")
                content_list.append({"type": "text", "text": "\n=== 🧐 INSPECTION TASK: TEST IMAGE ===\n"})
                content_list.append({"type": "image", "image": test_img, "_path": image_path})
            else: return "Error", "Image Not Found", []
        except: return "Error", "Load Fail", []

        options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
        task_text = f"\nQuestion: {question}\nOptions:\n{options_str}\n\nFinal Answer format: 'The correct answer is (X)'."
        content_list.append({"type": "text", "text": task_text})

        inference_list = []
        replay_history = []
        for item in content_list:
            clean_item = {k: v for k, v in item.items() if k != "_path"}
            inference_list.append(clean_item)
            if item['type'] == 'text': replay_history.append({"type": "text", "text": item['text']})
            elif item['type'] == 'image': replay_history.append({"type": "image", "path": item.get("_path", "memory")})

        response = self.vlm.generate(inference_list)
        
        match = re.search(r"The correct answer is \(?([A-D])\)?", response)
        pred = match.group(1) if match else "A"
        return pred, response, replay_history

    def solve(self, sample):
        image_path = sample["image_path"]
        subclass = sample["subclass"]
        question = sample["question"]
        options = sample["options"]
        
        rag = self._get_rag_engine(subclass)
        all_regions = [n for n, d in rag.G.nodes(data=True) if d.get("type") == "region"]

        # --- Build Content List ---
        content_list = []
        
        # A. System Instruction
        system_text = (
            f"You are an expert Quality Control Inspector for {subclass}.\n"
            f"I will provide you with the standard specifications and reference images for each region.\n"
            f"Please learn these standards strictly to identify any defects.\n\n"
        )
        content_list.append({"type": "text", "text": system_text})

        # B. RAG Loop
        for region in all_regions:
            info = rag.get_inspection_checklist(region)
            standard = info.get("normal_standard", "No specific standard.")
            
            defects = []
            for d in info.get("defects", []):
                try:
                    if "|" in d:
                        parts = d.split('|')
                        d_name = parts[0].replace("Type:", "").strip()
                        d_vis = parts[1].replace("Visual:", "").strip()
                        defects.append(f"{d_name}: {d_vis}")
                    else:
                        defects.append(d)
                except:
                    defects.append(str(d))

            region_text = (
                f"--- Reference Knowledge for Region: '{region}' ---\n"
                f"Standard: {standard}\n"
                f"Potential Defects: {'; '.join(defects)}\n"
            )
            content_list.append({"type": "text", "text": region_text})

            img_paths = rag.get_image_paths(region)
            if img_paths:
                selected_paths = img_paths[:self.k_shot]
                if selected_paths:
                    content_list.append({"type": "text", "text": f"Visual Reference (Normal Sample) for {region}:\n"})
                    for p in selected_paths:
                        try:
                            if os.path.exists(p):
                                ref_img = Image.open(p).convert("RGB")
                                content_list.append({
                                    "type": "image", 
                                    "image": ref_img, 
                                    "_path": p 
                                })
                        except Exception as e:
                            pass
            
            content_list.append({"type": "text", "text": "\n\n"})

        # C. Test Image
        try:
            if os.path.exists(image_path):
                test_img = Image.open(image_path).convert("RGB")
                content_list.append({"type": "text", "text": "--- Inspection Task ---\nBelow is the Test Image to inspect:\n"})
                content_list.append({
                    "type": "image", 
                    "image": test_img,
                    "_path": image_path
                })
            else:
                return "Error", "Image Not Found", []
        except Exception as e:
            return "Error", str(e), []

        # D. QA Task
        options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
        task_text = (
            f"\n### Question\n{question}\n"
            f"### Options\n{options_str}\n\n"
            f"### Execution Steps\n"
            f"1. Compare the Test Image against the Reference Knowledge for each region above.\n"
            f"2. Identify if any region deviates from the 'Standard' or matches a 'Defect'.\n"
            f"3. Select the correct option.\n"
            f"Final Answer format: 'The correct answer is (X)'."
        )
        content_list.append({"type": "text", "text": task_text})

        # Inference Prep
        inference_list = []
        replay_history = []
        
        for item in content_list:
            clean_item = {k: v for k, v in item.items() if k != "_path"}
            inference_list.append(clean_item)
            
            if item['type'] == 'text':
                replay_history.append({"type": "text", "text": item['text']})
            elif item['type'] == 'image':
                replay_history.append({"type": "image", "path": item.get("_path", "unknown")})

        # Run Inference
        response = self.vlm.generate(inference_list)

        # Parse Prediction
        match = re.search(r"The correct answer is \(?([A-D])\)?", response)
        if match:
            pred = match.group(1)
        else:
            candidates = re.findall(r"\b([A-D])\b", response)
            pred = candidates[-1] if candidates else "A"
        
        return pred, response, replay_history

# ======================= 3. Worker & Metrics =======================

def worker_entry(rank, gpu_ids, all_chunks, args, cache_root, temp_dir):
    gpu_id = gpu_ids[rank]
    device = f"cuda:{gpu_id}"
    samples = all_chunks[rank]
    
    txt_log_path = os.path.join(args.output_dir, f"debug_log_rank_{rank}.txt")
    print(f"[Worker {rank}] Init on GPU {gpu_id}...")
    
    try:
        vlm = LocalVLMInference(args.model_path, device, model_type=args.model_type)
        agent = SimInspecAgent(vlm, cache_root, k_shot=args.k_shot)
    except Exception as e:
        print(f"[Worker {rank}] Init Error: {e}")
        return

    full_logs = []

    with open(txt_log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Debug Session Started: {datetime.now()} ===\n")
        f.write(f"Task: Only Annotation=True\n\n")

    for i, sample in enumerate(tqdm(samples, desc=f"Worker {rank}")):
        try:
            pred, response, replay_history = agent.solve(sample)
            is_correct = (pred == sample["gt_answer"])
            
            # Write verbose text log
            with open(txt_log_path, "a", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write(f"🔹 ID: {sample['sample_id']} | Type: {sample['task_type']}\n")
                f.write(f"🔹 GT: {sample['gt_answer']} | Pred: {pred} | {'✅' if is_correct else '❌'}\n")
                f.write(f"🤖 Response:\n{response}\n\n")
            
            # Save structured log for metrics
            full_logs.append({
                "sample_id": sample["sample_id"],
                "task_type": sample["task_type"], # 关键字段用于统计
                "gt_answer": sample["gt_answer"],
                "prediction": pred,
                "is_correct": is_correct,
                "replay_history": replay_history
            })

        except Exception as e:
            print(f"Error processing {sample['sample_id']}: {e}")

    # Save JSON for aggregation
    json_log_path = os.path.join(args.output_dir, f"full_log_rank_{rank}.json")
    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(full_logs, f, indent=2, ensure_ascii=False)
    print(f"✅ [Worker {rank}] Logs saved.")


def calculate_metrics(output_dir):
    """
    读取所有生成的 JSON 日志，按 task_type 统计正确率
    """
    print("\n" + "="*60)
    print("📊 Calculating Per-Task Metrics...")
    print("="*60)
    
    json_files = glob.glob(os.path.join(output_dir, "full_log_rank_*.json"))
    if not json_files:
        print("❌ No log files found.")
        return

    # 统计容器： { 'Anomaly Detection': {'total': 100, 'correct': 80}, ... }
    stats = {} 
    grand_total = 0
    grand_correct = 0

    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # 获取任务类型，如果为空则归为 "Unknown"
                t_type = item.get("task_type", "Unknown")
                is_cor = item.get("is_correct", False)
                
                if t_type not in stats:
                    stats[t_type] = {'total': 0, 'correct': 0}
                
                stats[t_type]['total'] += 1
                if is_cor:
                    stats[t_type]['correct'] += 1
                
                grand_total += 1
                if is_cor: grand_correct += 1

    # 打印表头
    print(f"{'Task Type':<30} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<10}")
    print("-" * 65)
    
    # 打印各类别的统计
    for t_type, counts in sorted(stats.items()):
        total = counts['total']
        correct = counts['correct']
        acc = (correct / total) * 100 if total > 0 else 0.0
        print(f"{t_type:<30} | {total:<8} | {correct:<8} | {acc:.2f}%")
    
    print("-" * 65)
    
    # 打印总计
    overall_acc = (grand_correct / grand_total) * 100 if grand_total > 0 else 0.0
    print(f"{'OVERALL (Weighted)':<30} | {grand_total:<8} | {grand_correct:<8} | {overall_acc:.2f}%")
    print("=" * 65 + "\n")


def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    database_root = kragad_paths.database_root()
    cache_root = args.graph_cache_root or os.path.join(database_root, "graph_index")
    
    print(f"[System] Loading dataset for subclass: {args.subclass}...")
    dataset = DSMVTecDatasetLoader(args.dataset_root)
    # Parse samples
    raw_samples = list(dataset.parse_samples(args.subclass))
    
    target_samples = []
    
    # Debug Range Filtering
    start_idx = -1
    end_idx = -1
    if args.range:
        try:
            parts = args.range.split("-")
            start_idx = int(parts[0])
            end_idx = int(parts[1]) if len(parts) > 1 else start_idx
            print(f"🎯 [Debug Mode] Range: {start_idx} - {end_idx}")
        except: pass

    # Filtering Logic
    for i, s in enumerate(raw_samples):
        # 1. Annotation Filter (Crucial!)
        if not s.get("annotation", False):
            continue

        # 2. Range Filter
        filename = os.path.basename(s["image_path"])
        match = re.search(r"(\d+)", filename)
        if match:
            img_idx = int(match.group(1))
            if start_idx != -1:
                if not (start_idx <= img_idx <= end_idx):
                    continue
            
            s["sample_id"] = f"{args.subclass}_{img_idx}_{i}" 
            target_samples.append(s)

    print(f"[System] Selected {len(target_samples)} valid annotated samples.")
    if len(target_samples) == 0: return

    # Distributed Setup
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    chunk_size = math.ceil(len(target_samples) / len(gpu_ids))
    chunks = [target_samples[i:i + chunk_size] for i in range(0, len(target_samples), chunk_size)]
    while len(chunks) < len(gpu_ids): chunks.append([])

    # Run Workers
    mp.spawn(
        worker_entry,
        args=(gpu_ids, chunks, args, cache_root, temp_dir),
        nprocs=len(gpu_ids),
        join=True
    )
    
    # Calculate Final Metrics
    calculate_metrics(args.output_dir)

if __name__ == "__main__":
    try: mp.set_start_method('spawn')
    except RuntimeError: pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=kragad_paths.vlm_model_path("qwen2d5-vl-7B"), help="Local model path")
    parser.add_argument("--dataset_root", default=os.path.join(kragad_paths.dataset_root(), "DS-MVTec"), help="Dataset root")
    parser.add_argument("--output_dir", default="./results_metrics", help="Log output directory")
    parser.add_argument("--graph_cache_root", default=None,
                        help="Graph index directory. Defaults to $KRAGAD_DATABASE_ROOT/graph_index.")
    parser.add_argument("--subclass", type=str, default="cable", help="cable or bottle")
    parser.add_argument("--gpus", type=str, default="0", help="GPU ID")
    parser.add_argument("--range", type=str, default=None, help="Debug range (e.g., '000-002')")
    parser.add_argument("--k_shot", type=int, default=1, help="Number of ref images per region")
    parser.add_argument("--model_type", type=str, default="qwen2.5")
    
    args = parser.parse_args()
    evaluate(args)
