import os
import json
import argparse
# 请确保你的文件名是 GraphRag.py，且与本脚本在同一目录下
from kragad import paths as kragad_paths
from kragad.rag.GraphRag import SimInspecGraphEngine  

# ================= 配置 =================
PROJECT_ROOT = kragad_paths.project_root()
DATA_ROOT = kragad_paths.data_root()
DATABASE_ROOT = kragad_paths.database_root()
MAX_K_SHOT = 4

def get_paths(dataset_name: str, text_kb_root: str = None, graph_output: str = None):
    """
    输入路径区分 dataset；默认输出保持原逻辑。
    可通过参数指定新的 text knowledge 根目录和 graph 输出目录，用于消融/敏感性实验。
    """
    d_name = dataset_name.lower() 
    if text_kb_root is None:
        text_kb_root = os.path.join(DATABASE_ROOT, "text_knowledge")
    if graph_output is None:
        graph_output = os.path.join(DATABASE_ROOT, "graph_index")
    return {
        # 输入：区分 mvtec / visa
        "text_kb": os.path.join(text_kb_root, d_name),
        "img_source": os.path.join(DATABASE_ROOT, "img", d_name),

        # 输出：默认保持原逻辑，统一放在 graph_index 根目录
        "graph_output": graph_output
    }

def find_reference_images(img_source_root: str, category: str, region_key: str) -> list:
    """自动查找 K-Shot 图片 (兼容 000 和 0000 文件夹命名)"""
    found_images = []
    base_path = os.path.join(img_source_root, category)
    
    # 遍历前 MAX_K_SHOT 个样本
    for i in range(MAX_K_SHOT):
        # 1. 优先尝试 4 位命名 (VisA常见: 0000)
        shot_folder_4 = os.path.join(base_path, f"{i:04d}")
        target_img_4 = os.path.join(shot_folder_4, f"{region_key}.png")
        
        # 2. 其次尝试 3 位命名 (MVTec常见: 000)
        shot_folder_3 = os.path.join(base_path, f"{i:03d}")
        target_img_3 = os.path.join(shot_folder_3, f"{region_key}.png")

        if os.path.exists(target_img_4):
            found_images.append(target_img_4)
        elif os.path.exists(target_img_3):
            found_images.append(target_img_3)
            
    return found_images

def build_process(dataset_name: str, category: str, text_kb_root: str = None, graph_output: str = None):
    paths = get_paths(dataset_name, text_kb_root=text_kb_root, graph_output=graph_output)
    
    print(f"\n====== Starting Build Process for: [{dataset_name}] -> [{category}] ======")
    
    # 1. 初始化
    engine = SimInspecGraphEngine() 
    
    # 2. 加载 JSON
    json_path = os.path.join(paths["text_kb"], f"{category}.json")
    if not os.path.exists(json_path):
        print(f"[Error] JSON file not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[Error] Invalid JSON format in {category}.json: {e}")
            return
    
    # 3. 构建语义图谱
    try:
        engine.load_json(json_data)
    except Exception as e:
        print(f"[Error] Failed to load JSON into Graph Engine: {e}")
        return
    
    # 4. 注入多模态图片路径
    print(f"[*] Injecting Multimodal Paths for {category}...")
    
    count_imgs = 0
    for node_name in engine.G.nodes():
        node_type = engine.G.nodes[node_name].get("type")
        
        if node_type == "region":
            # 使用动态获取的 img_source 路径查找
            imgs = find_reference_images(paths["img_source"], category, node_name)
            
            if imgs:
                engine.G.nodes[node_name]['image_paths'] = imgs
                count_imgs += len(imgs)
                print(f"    -> Region '{node_name}': Linked {len(imgs)} images.")
            else:
                print(f"    [Warning] No images found for region '{node_name}'")

    # 5. 保存 (保持完全原有的逻辑)
    if not os.path.exists(paths["graph_output"]):
        os.makedirs(paths["graph_output"])
        
    # 文件名保持原样: {category}_graph.pkl
    output_path = os.path.join(paths["graph_output"], f"{category}_graph.pkl")
    engine.save_to_disk(output_path)
    print(f"✅ Build Success! Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 必须指定 dataset 来确定输入路径
    parser.add_argument("--dataset", "-d", type=str, required=True, 
                        choices=["mvtec", "visa", "MVTEC", "VISA"],
                        help="Target dataset name (mvtec or visa)")
    
    parser.add_argument("--category", "-c", type=str, required=False, 
                        help="Specific category to build. If omitted, builds ALL found in text_kb.")
    parser.add_argument("--text_kb_root", type=str, default=None,
                        help="Root directory that contains dataset folders, e.g. $GLLS_DATABASE_ROOT/text_knowledge_weak_agent")
    parser.add_argument("--graph_output", type=str, default=None,
                        help="Output directory for graph pkl files, e.g. $GLLS_DATABASE_ROOT/graph_index_weak_agent")
    
    args = parser.parse_args()
    target_dataset = args.dataset.lower()
    
    if args.category:
        # 单个构建
        build_process(target_dataset, args.category, text_kb_root=args.text_kb_root, graph_output=args.graph_output)
    else:
        # 批量构建
        paths = get_paths(target_dataset, text_kb_root=args.text_kb_root, graph_output=args.graph_output)
        kb_dir = paths["text_kb"]
        
        if not os.path.exists(kb_dir):
            print(f"[Fatal Error] Text knowledge directory does not exist: {kb_dir}")
            exit(1)
            
        json_files = [f for f in os.listdir(kb_dir) if f.endswith(".json")]
        
        if not json_files:
            print(f"[Warning] No .json files found in {kb_dir}")
        else:
            print(f"[*] Found {len(json_files)} knowledge bases for [{target_dataset}]. Starting batch build...")
            for json_file in json_files:
                category_name = os.path.splitext(json_file)[0]
                try:
                    build_process(target_dataset, category_name, text_kb_root=args.text_kb_root, graph_output=args.graph_output)
                except Exception as e:
                    print(f"[Error] Unexpected error processing {category_name}: {e}")
