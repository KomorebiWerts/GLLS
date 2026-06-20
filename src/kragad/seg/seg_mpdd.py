import os
import argparse
import numpy as np
import cv2
from kragad.seg.sam3_engine import Sam3Engine
from kragad import paths as kragad_paths

# ================= 配置 =================
DEFAULT_CHECKPOINT = kragad_paths.sam3_path()

# 路径配置
BASE_DATA_DIR = kragad_paths.mpdd_root()
OUTPUT_ROOT = kragad_paths.mpdd_root()

CATEGORY_CONFIG = {
    "bracket_black": {"prompts": "the black object"},
    "bracket_brown": {"prompts": "the brown object"},
    "bracket_white": {"prompts": "the white object"},
    "connector":     {"prompts": "small metal tightening clamp"},
    "metal_plate":   {"prompts": "the metal plate"},
    "tubes":         {"prompts": "the metal tubes"}
}

# ================= 处理逻辑 =================

def process_and_save(engine, img_path, save_path, prompt, keep_holes=False):
    """
    核心处理函数：支持多物体保留
    """
    engine.set_image(img_path)
    
    # 1. 尝试文本提示
    mask_all, _ = engine.predict_mask(prompt,threshold=0.2)
    
    # 2. 兜底策略：如果文本失败，使用中心框
    if mask_all is None:
        print(f"    [Warn] Text prompt failed for {os.path.basename(img_path)}, trying Center Box...")
        center_box = [[0.1, 0.1, 0.8, 0.8]] 
        mask_all, _ = engine.predict_mask_with_boxes(center_box)

    if mask_all is None:
        print(f"    [Error] Object not detected: {os.path.basename(img_path)}")
        return

    # 转换为 uint8 0/255
    mask_uint8 = (mask_all.astype(np.uint8) * 255)
    
    # 创建一个空的黑底图用于绘制最终结果
    mask_final = np.zeros_like(mask_uint8)

    # 面积阈值：过滤掉太小的噪点 (例如小于100像素的斑点)
    AREA_THRESHOLD = 100

    # 后处理
    if not keep_holes:
        # --- 实心物体 (Bracket, Connector) ---
        # 需求：保留所有独立的物体，但填补每个物体内部的孔洞
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            count = 0
            for contour in contours:
                # 【修改点】遍历所有轮廓，而不是只取 max
                if cv2.contourArea(contour) > AREA_THRESHOLD:
                    # 将该轮廓内部填充满 (thickness=cv2.FILLED)
                    cv2.drawContours(mask_final, [contour], -1, 255, thickness=cv2.FILLED)
                    count += 1
            if count == 0: # 如果所有轮廓都太小，回退到原始mask
                 mask_final = mask_uint8
        else:
            mask_final = mask_uint8
            
    else:
        # --- 中空物体 (Metal plate, Tubes) ---
        # 需求：保留所有独立的物体，且保留它们内部原本的孔洞
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        if num_labels > 1:
            # label 0 是背景，从 1 开始遍历
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                # 【修改点】保留所有足够大的连通域
                if area > AREA_THRESHOLD:
                    mask_final[labels == i] = 255
        else:
            mask_final = mask_uint8

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, mask_final)
    print(f"    [Saved] {save_path}")

# ================= 主程序 =================

def main():
    parser = argparse.ArgumentParser(description="MPDD Mask Generator")
    parser.add_argument("--category", "-c", type=str, default="all", help="Category name or 'all'")
    parser.add_argument("--input_dir", type=str, default=BASE_DATA_DIR, help="MPDD Root Dir")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_ROOT, help="Where to save fg_mask folder")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT, help="SAM3 checkpoint")

    args = parser.parse_args()

    # 初始化 SAM3
    try:
        engine = Sam3Engine(args.ckpt)
    except Exception as e:
        print(f"Failed to load SAM3: {e}")
        return

    if args.category == 'all':
        target_cats = CATEGORY_CONFIG.keys()
    else:
        target_cats = [args.category]

    print(f"\n=== Generating Masks for MPDD (First 4 images, Multiple Objects) ===")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {os.path.join(args.output_dir, 'fg_mask')}")

    for cat in target_cats:
        if cat not in CATEGORY_CONFIG: continue
        
        print(f"\n--- Processing {cat} ---")
        prompt = CATEGORY_CONFIG[cat]["prompts"]
        
        # 策略：metal_plate 和 tubes 需要保留孔洞
        keep_holes = cat in ["metal_plate", "tubes"]

        img_dir = os.path.join(args.input_dir, cat, "train", "good")
        if not os.path.exists(img_dir):
            print(f"    [Skip] Path not found: {img_dir}")
            continue

        save_dir_base = os.path.join(args.output_dir, "fg_mask", cat)
        
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))]
        images.sort()

        # 依然只处理前4张 (便于调试，如果需要全部处理请注释掉这行)
        images = images[:30]
        print(f"    Processing subset: {len(images)} images")

        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            save_path = os.path.join(save_dir_base, img_name)
            
            process_and_save(engine, img_path, save_path, prompt, keep_holes)

    print("\nDone. Check the 'fg_mask' folder.")

if __name__ == "__main__":
    main()
