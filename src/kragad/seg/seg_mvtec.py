
# ================= 主程序 =================
import os
import argparse
import numpy as np
import shutil  # 新增：用于直接复制文件
from kragad.seg.sam3_engine import Sam3Engine
from kragad import paths as kragad_paths
from scipy.ndimage import binary_fill_holes 
import cv2

# ================= 默认配置 =================
DEFAULT_CHECKPOINT = kragad_paths.sam3_path()
KRAGAD_DATABASE_ROOT = kragad_paths.database_root()
BASE_DATA_DIR = os.path.join(KRAGAD_DATABASE_ROOT, "fig", "DS_MVTec")
OUTPUT_ROOT = os.path.join(KRAGAD_DATABASE_ROOT, "img", "mvtec")

# ================= 类别策略配置 =================
CATEGORY_CONFIG = {
    # --- 复杂逻辑类 (需要 SAM3 切割) ---
    "bottle": {
        "prompts": { "whole": "the entire dark glass bottle object", "inner": "the circular hole in the very center" },
        "demo_images": ["000.png", "001.png", "002.png", "003.png"]
    },
    "capsule": {
        "prompts": { "whole": "the entire capsule pill object", "part_left": "the dark black half of the capsule" },
        "demo_images": ["000.png", "001.png", "002.png", "003.png"]
    },
    "cable": {
        "prompts": { "outer_ring": "the large white circular object", 
                    "copper_cores": "the inner copper strands inside the wire" 
                    },
        "demo_images": ["000.png", "001.png", "002.png", "003.png"]
    },
    "screw": {
        "prompts": { "whole": "the entire metal screw object", "thread": "the ridged and jagged section of the screw shaft" },
        "demo_images": ["000.png", "001.png", "002.png", "003.png"]
    },
    "transistor": {
        "prompts": { "body": "the black rectangular body", "legs": "the silver metal legs extending from the black body" },
        "demo_images": ["000.png", "001.png", "002.png", "003.png"]
    },
    "zipper": {
        "prompts": { "whole": "the entire black zipper tape object", "teeth": "the central interlocking zipper teeth" },
        "demo_images": ["000.png", "001.png", "002.png", "003.png"]
    },
    "toothbrush": {
        "prompts": { 
            "whole": "Toothbrush", 
        },
        "demo_images": ["000.png", "001.png", "002.png", "003.png"]
    },
    "pill": {
        "prompts": { 
            "whole": "the pill", 
        },
        "demo_images": ["000.png", "001.png", "002.png", "003.png"]
    },
    "metal_nut": {
        "prompts": { 
            "whole": "the green part of the Metal_nut only", 
        },
        "demo_images": ["000.png", "001.png", "002.png", "003.png"]
    },
    # --- 简单逻辑类 (直接原图 copy) ---
    # 提示词在这里只是占位符，不会被调用
    "hazelnut":   {"prompts": {}, "demo_images": ["000.png", "001.png", "002.png", "003.png"]},
    "carpet":     {"prompts": {}, "demo_images": ["000.png", "001.png", "002.png", "003.png"]},
    "grid":       {"prompts": {}, "demo_images": ["000.png", "001.png", "002.png", "003.png"]},
    "leather":    {"prompts": {}, "demo_images": ["000.png", "001.png", "002.png", "003.png"]},
    "tile":       {"prompts": {}, "demo_images": ["000.png", "001.png", "002.png", "003.png"]},
    "wood":       {"prompts": {}, "demo_images": ["000.png", "001.png", "002.png", "003.png"]}
}

def process_pill(engine, img_name, save_dir, prompts, full_img_path):
    print(f"  [Logic] Processing pill (SAM3): {img_name}")
    
    # 1. 切割整体 (Standard Whole Object)
    mask_whole, _ = engine.predict_mask(prompts["whole"])
    if mask_whole is None: return
    engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "part_surface.png"))
    

def process_metal_nut(engine, img_name, save_dir, prompts, full_img_path):
    print(f"  [Logic] Processing metal_nut (SAM3): {img_name}")
    
    # 1. 切割整体 (Standard Whole Object)
    mask_whole, _ = engine.predict_mask(prompts["whole"])
    if mask_whole is None: return
    engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "part_surface.png"))

# ================= 处理函数定义 =================
def process_toothbrush(engine, img_name, save_dir, prompts, full_img_path):
    print(f"  [Logic] Processing Toothbrush (SAM3): {img_name}")
    
    # 1. 切割整体 (Standard Whole Object)
    mask_whole, _ = engine.predict_mask(prompts["whole"])
    if mask_whole is None: return
    engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "part_bristles.png"))


def process_zipper(engine, img_name, save_dir, prompts, full_img_path):
    print(f"  [Logic] Processing Zipper (SAM3): {img_name}")
    mask_whole, _ = engine.predict_mask(prompts["whole"])
    if mask_whole is None: return
    engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "whole_object.png"))

    mask_teeth, _ = engine.predict_mask(prompts["teeth"])
    if mask_teeth is None: mask_teeth = np.zeros_like(mask_whole, dtype=bool)
    else: mask_teeth = np.logical_and(mask_teeth, mask_whole)
    engine.save_masked_cutout(mask_teeth, os.path.join(save_dir, "part_zipper.png"))

    mask_fabric = np.logical_and(mask_whole, np.logical_not(mask_teeth))
    if np.any(mask_fabric):
        engine.save_masked_cutout(mask_fabric, os.path.join(save_dir, "part_fabric.png"))
        
def process_transistor(engine, img_name, save_dir, prompts, full_img_path):
    print(f"  [Logic] Processing Transistor (SAM3): {img_name}")
    mask_body, _ = engine.predict_mask(prompts["body"])
    mask_legs_raw, _ = engine.predict_mask(prompts["legs"])
    if mask_body is None: mask_body = np.zeros((1024, 1024), dtype=bool)
    
    mask_legs_roi = np.zeros_like(mask_body, dtype=np.uint8)
    h, w = mask_body.shape
    x1, x2, y1, y2 = 0, w, 0, h
    valid_roi = False

    if mask_legs_raw is not None and np.any(mask_legs_raw):
        y_indices, x_indices = np.where(mask_legs_raw)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        pad_x, pad_y_bottom, pad_y_top = 50, 40, 10
        x1, x2 = max(0, x_min - pad_x), min(w, x_max + pad_x)
        y1, y2 = max(0, y_min - pad_y_top), min(h, y_max + pad_y_bottom)
        valid_roi = True
    elif np.any(mask_body):
        y_indices, x_indices = np.where(mask_body)
        y_body_bottom = np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        x1, x2 = max(0, x_min - 50), min(w, x_max + 50)
        y1, y2 = y_body_bottom - 5, min(h, y_body_bottom + 250)
        valid_roi = True

    if valid_roi: mask_legs_roi[y1:y2, x1:x2] = 1
    mask_legs_roi_bool = mask_legs_roi.astype(bool)
    mask_whole = np.logical_or(mask_body, mask_legs_roi_bool)

    if np.any(mask_body): engine.save_masked_cutout(mask_body, os.path.join(save_dir, "part_body.png"))
    if np.any(mask_legs_roi_bool): engine.save_masked_cutout(mask_legs_roi_bool, os.path.join(save_dir, "part_legs.png"))
    if np.any(mask_whole): engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "whole_object.png"))

def process_screw(engine, img_name, save_dir, prompts, full_img_path):
    print(f"  [Logic] Processing Screw (SAM3): {img_name}")
    mask_whole, _ = engine.predict_mask(prompts["whole"])
    if mask_whole is None: return
    engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "whole_object.png"))

    mask_thread, _ = engine.predict_mask(prompts["thread"])
    if mask_thread is not None:
        mask_thread = np.logical_and(mask_thread, mask_whole)
        engine.save_masked_cutout(mask_thread, os.path.join(save_dir, "part_thread.png"))
        
        kernel = np.ones((5, 5), np.uint8)
        dilated_thread_mask = cv2.dilate(mask_thread.astype(np.uint8)*255, kernel, iterations=2).astype(bool)
        mask_rest = np.logical_and(mask_whole, np.logical_not(dilated_thread_mask))
        if np.any(mask_rest):
            engine.save_masked_cutout(mask_rest, os.path.join(save_dir, "part_head_shank.png"))
    else:
        engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "part_head_shank.png"))
        
# def process_cable(engine, img_name, save_dir, prompts, full_img_path):
#     print(f"  [Logic] Processing Cable (SAM3 - Simplified): {img_name}")
    
#     # 1. 预测外层白色圆环 (Outer Sheath)
#     mask_sheath, _ = engine.predict_mask(prompts["outer_ring"])
#     if mask_sheath is None: 
#         return

#     # 2. 生成整体掩码 (Whole Object)
#     # 逻辑：找到外皮最大的轮廓，并填充其内部，这样就包含了内部的绝缘体和铜线
#     sheath_uint8 = mask_sheath.astype(np.uint8) * 255
#     contours_sheath, _ = cv2.findContours(sheath_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     mask_whole = np.zeros_like(mask_sheath, dtype=np.uint8)
#     if contours_sheath:
#         # 找到最大轮廓并填充内部 (Thickness=-1/FILLED)
#         cv2.drawContours(mask_whole, [max(contours_sheath, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    
#     mask_whole_bool = mask_whole.astype(bool)

#     # 3. 保存整体和外皮
#     engine.save_masked_cutout(mask_whole_bool, os.path.join(save_dir, "whole_object.png"))
#     engine.save_masked_cutout(mask_sheath, os.path.join(save_dir, "part_sheath.png"))

#     # 4. 计算并保存内部核心 (Inner Core = Whole - Sheath)
#     # 这里包含了绝缘环和铜线，不再细分
#     mask_inner_core = np.logical_and(mask_whole_bool, np.logical_not(mask_sheath))
    
#     if np.any(mask_inner_core):
#         # 命名为 part_insulation.png 或 part_core.png 均可，这里沿用 insulation 代表内部结构
#         engine.save_masked_cutout(mask_inner_core, os.path.join(save_dir, "part_insulation.png"))

def process_cable(engine, img_name, save_dir, prompts, full_img_path):
    print(f"  [Logic] Processing Cable (SAM3): {img_name}")
    mask_copper, _ = engine.predict_mask(prompts["copper_cores"])
    mask_sheath, _ = engine.predict_mask(prompts["outer_ring"])
    if mask_sheath is None: return

    sheath_uint8 = mask_sheath.astype(np.uint8) * 255
    contours_sheath, _ = cv2.findContours(sheath_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_whole = np.zeros_like(mask_sheath, dtype=np.uint8)
    if contours_sheath:
        cv2.drawContours(mask_whole, [max(contours_sheath, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    mask_whole_bool = mask_whole.astype(bool)

    engine.save_masked_cutout(mask_whole_bool, os.path.join(save_dir, "whole_object.png"))
    engine.save_masked_cutout(mask_sheath, os.path.join(save_dir, "part_sheath.png"))

    if mask_copper is None: mask_copper = np.zeros_like(mask_sheath, dtype=bool)
    else: mask_copper = np.logical_and(mask_copper, mask_whole_bool)
    if np.any(mask_copper): engine.save_masked_cutout(mask_copper, os.path.join(save_dir, "part_copper.png"))

    mask_insulation = np.logical_and(mask_whole_bool, np.logical_not(mask_sheath))
    mask_insulation = np.logical_and(mask_insulation, np.logical_not(mask_copper))
    if np.any(mask_insulation): engine.save_masked_cutout(mask_insulation, os.path.join(save_dir, "part_insulation.png"))

def process_bottle(engine, img_name, save_dir, prompts, full_img_path):
    print(f"  [Logic] Processing Bottle (SAM3): {img_name}")
    mask_whole, _ = engine.predict_mask(prompts["whole"])
    if mask_whole is None: return
    engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "whole_object.png"))

    mask_inner, _ = engine.predict_mask(prompts["inner"])
    if mask_inner is None:
        engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "surface_rest.png"))
    else:
        engine.save_masked_cutout(mask_inner, os.path.join(save_dir, "interior_void.png"))
        mask_rest = np.logical_and(mask_whole, np.logical_not(mask_inner))
        engine.save_masked_cutout(mask_rest, os.path.join(save_dir, "surface_rest.png"))

def process_capsule(engine, img_name, save_dir, prompts, full_img_path):
    print(f"  [Logic] Processing Capsule (SAM3): {img_name}")
    mask_whole, _ = engine.predict_mask(prompts["whole"])
    if mask_whole is None: return
    engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "whole_object.png"))

    mask_left_raw, _ = engine.predict_mask(prompts["part_left"])
    if mask_left_raw is not None:
        mask_left_clean = np.logical_and(mask_whole, mask_left_raw)
        engine.save_masked_cutout(mask_left_clean, os.path.join(save_dir, "part_cap.png"))
        y_indices, x_indices = np.where(mask_left_clean)
        if len(x_indices) > 0:
            split_x = np.max(x_indices) + 2
            mask_right = mask_whole.copy()
            mask_right[:, :split_x] = False
            if np.any(mask_right):
                engine.save_masked_cutout(mask_right, os.path.join(save_dir, "part_body.png"))

# ================= 核心修改：直接复制函数 =================
def process_direct_copy(engine, img_name, save_dir, prompts, full_img_path):
    """
    不调用 SAM3，直接将原始图片复制到目标目录，并命名为 whole_object.png
    """
    print(f"  [Logic] Direct Copy (No SAM3): {img_name}")
    
    # 确保目标文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dst_path = os.path.join(save_dir, "whole_object.png")
    
    try:
        shutil.copy2(full_img_path, dst_path)
        print(f"    [Success] Copied to: {dst_path}")
    except Exception as e:
        print(f"    [Error] Copy failed: {e}")

# ================= 动态函数映射 =================
# 将剩下9个类的处理函数都指向 process_direct_copy
process_hazelnut = process_direct_copy
process_carpet = process_direct_copy
process_grid = process_direct_copy
process_leather = process_direct_copy
process_tile = process_direct_copy
process_wood = process_direct_copy

# ================= 主程序 (最终修改版) =================
def main():
    parser = argparse.ArgumentParser(description="MVTec Processing Runner")
    # 修改说明：支持 'simple' 模式
    parser.add_argument("--category", "-c", type=str, required=True, 
                        help="Category name, or 'simple' to run all 9 direct-copy classes.")
    parser.add_argument("--input_dir", type=str, default=BASE_DATA_DIR, help="Base directory")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_ROOT, help="Root output directory")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT, help="SAM3 checkpoint")

    args = parser.parse_args()

    # === 定义那9个不需要SAM3的简单类别 ===
    SIMPLE_CATEGORIES = [
        "hazelnut", 
        "carpet", "grid", "leather", "tile", "wood"
    ]

    # 1. 确定要处理的任务列表
    target_list = []
    
    if args.category.lower() == 'simple':
        print(f"🚀 Mode: [SIMPLE BATCH]. Will process 9 categories: {SIMPLE_CATEGORIES}")
        target_list = SIMPLE_CATEGORIES
        # 标记：不需要加载引擎
        need_engine = False
    elif args.category.lower() == 'all':
        # 如果你以后想跑全部15个类
        target_list = sorted(list(CATEGORY_CONFIG.keys()))
        need_engine = True # 包含复杂类，必须加载
    else:
        # 单类模式
        if args.category.lower() not in CATEGORY_CONFIG:
            print(f"❌ Error: Category '{args.category}' not found.")
            return
        target_list = [args.category.lower()]
        # 只有当这个类不在简单列表中时，才需要引擎
        need_engine = args.category.lower() not in SIMPLE_CATEGORIES

    # 2. 按需加载 SAM3 引擎 (省时省显存)
    engine = None
    if need_engine:
        print("\n🔧 Initializing SAM3 Engine (Required for complex tasks)...")
        try:
            engine = Sam3Engine(args.ckpt)
            print("✅ SAM3 Engine loaded.")
        except Exception as e:
            print(f"❌ Failed to load engine: {e}")
            return
    else:
        print("\n⏩ Skipping SAM3 Engine initialization (Direct copy mode).")

    # 3. 批量循环处理
    for i, cat_name in enumerate(target_list):
        print(f"\n[{i+1}/{len(target_list)}] Processing: {cat_name.upper()} " + "-"*20)
        
        input_path = os.path.join(args.input_dir, cat_name)
        
        # 检查输入路径是否存在
        if not os.path.exists(input_path):
            print(f"⚠️ Input path not found: {input_path}, Skipping.")
            continue
            
        # 获取配置
        cat_conf = CATEGORY_CONFIG[cat_name]
        image_list = cat_conf.get("demo_images")
        image_list = [f for f in image_list if os.path.exists(os.path.join(input_path, f))]
        image_list.sort()
        
        if not image_list:
            print("⚠️ No valid demo images found.")
            continue

        # 处理每一张图
        for img_name in image_list:
            full_img_path = os.path.join(input_path, img_name)
            
            # 只有引擎存在时才设置图片（避免None报错）
            if engine:
                engine.set_image(full_img_path)
            
            # 准备输出目录
            current_save_dir = os.path.join(args.output_dir, cat_name, os.path.splitext(img_name)[0])
            
            # 获取处理函数
            func_name = f"process_{cat_name}"
            processor_func = globals().get(func_name)
            
            if processor_func:
                # 执行处理
                processor_func(engine, img_name, current_save_dir, cat_conf["prompts"], full_img_path)
            else:
                print(f"❌ No function defined for {cat_name}")

    print("\n✨ All tasks completed.")

if __name__ == "__main__":
    main()
