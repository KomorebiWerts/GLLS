import os
import argparse
import numpy as np
import cv2
from kragad.seg.sam3_engine import Sam3Engine
from kragad import paths as kragad_paths

# ================= 默认配置 =================
DEFAULT_CHECKPOINT = kragad_paths.sam3_path()
GLLS_DATABASE_ROOT = kragad_paths.database_root()
BASE_DATA_DIR = os.path.join(GLLS_DATABASE_ROOT, "fig", "VisA")
OUTPUT_ROOT = os.path.join(GLLS_DATABASE_ROOT, "img", "visa")

# ================= 类别策略配置 =================
CATEGORY_CONFIG = {
    "pcb1": {
        "prompts": {
            "whole_object": "The entire blue circuit board, including silver cylinders and metal pins",
            "cylinders": "cylinders"
        },
        "demo_images": ["0000.JPG", "0001.JPG", "0002.JPG", "0003.JPG"]
    },
    "pcb2": {
        "prompts": {
            "whole_board": "the entire circuit board, including all electronic components"
        },
        # 假设你的图片文件名是这些，如果不是请修改
        "demo_images": ["0000.JPG", "0001.JPG", "0002.JPG", "0003.JPG"] 
    },
    "pcb3": {
        "prompts": {
            "whole_object": "the circuit with metal and bulb",
        },
        "demo_images": ["0000.JPG", "0001.JPG", "0002.JPG", "0003.JPG"] # 请根据实际文件名修改
    },
    "pcb4": {
        "prompts": { 
            "whole_board": "the entire circuit board, including all electronic components" 
        },
        "demo_images": ["0000.JPG", "0001.JPG", "0002.JPG", "0003.JPG"]
    },
    "candle": {
        "prompts": {
            "whole_object": "all the round tea light candles",
            "wick": "the small white wick threads in the center of the tea light candles" 
        },
        "demo_images": ["0000.png", "0001.png", "0002.png", "0003.png"]
    },
    "capsules": {
        "prompts": {
            "target_object": "the capsules." 
        },
        "demo_images": ["000.JPG", "001.JPG", "002.JPG", "003.JPG"]
    },
    "cashew": {
        "prompts": {
            "target_object": "the kidney-shaped nut" 
        },
        "demo_images": ["000.JPG", "001.JPG", "002.JPG", "003.JPG"]
    },
    "chewinggum": {
        "prompts": {
            "target_object": "the white object" 
        },
        "demo_images": ["000.JPG", "001.JPG", "002.JPG", "003.JPG"]
    },
    "fryum": {
        "prompts": {
            "target_object": "light orange wheel" 
        },
        "demo_images": ["000.JPG", "001.JPG", "002.JPG", "003.JPG"]
    },
    "macaroni1": {
        "prompts": {
            "target_object": "orange elbow macaroni" 
        },
        "demo_images": ["0000.JPG", "0001.JPG", "0002.JPG", "0003.JPG"]
    },
    "macaroni2": {
        "prompts": {
            "target_object": "yellow elbow macaroni" 
        },
        "demo_images": ["0000.JPG", "0001.JPG", "0002.JPG", "0003.JPG"]
    },
    "pipe_fryum": {
        "prompts": {
            "target_object": "the pipe fryum" 
        },
        "demo_images": ["000.JPG", "001.JPG", "002.JPG", "003.JPG"]
    }
}

# ================= 辅助函数 =================
def crop_and_save(image, x1, y1, x2, y2, save_path):
    """
    根据坐标切割图片并保存 (用于 PCB)
    """
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    if x2 > x1 and y2 > y1:
        crop = image[y1:y2, x1:x2]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, crop)
        return True
    return False

def get_bbox_from_mask(mask):
    """从 Mask 获取 Bounding Box"""
    if mask is None or np.sum(mask) == 0: return None
    y_indices, x_indices = np.where(mask)
    return (np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices))
def process_hollow_object(engine, img_name, save_dir, prompt):
    """
    通用保孔逻辑：适用于 Fryum, Macaroni 等中空物体。
    直接取最大连通域，不使用 drawContours(FILLED)，从而保留孔洞。
    """
    print(f"  [Logic] Processing Hollow Object (Keep Holes): {img_name}")
    
    # 1. 预测 Mask
    mask_all, _ = engine.predict_mask(prompt)
    if mask_all is None: return

    # 2. 连通域分析 (找独立色块)
    # num_labels: 连通块数量(含背景), labels: 标签图, stats: 统计信息(含面积)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_all.astype(np.uint8), connectivity=8)
    
    if num_labels < 2: return # 只有背景

    # 3. 取最大连通块 (排除 label 0 背景)
    # stats[1:, 4] 是除了背景外的所有区域面积，argmax 找到最大值的索引
    best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # 4. 生成 Mask (保留了原始预测中的孔洞)
    mask_single = (labels == best_label).astype(bool)

    # 5. 保存
    engine.save_masked_cutout(mask_single, os.path.join(save_dir, "whole_object.png"))
    print("    [Saved] whole_object.png")

# --- 各类别接口函数 ---

def process_candle(engine, img_name, save_dir, prompts, full_img_path):
    """
    candle: 输出三个部分
      1) whole_object.png  -> 四个蜡烛集合（mask_all）
      2) part_wick.png     -> 单个蜡烛的 wick（在单蜡烛 mask 内取最大连通域）
      3) wax_surface.png   -> 单个蜡烛蜡面（single_candle - wick）
    prompt 只用两个：
      - prompts["whole_object"]
      - prompts["wick"]
    """
    print(f"  [Logic] Processing Candle (whole_object + wick + wax_surface): {img_name}")

    # --- 1) 预测四个蜡烛的集合 mask ---
    whole_prompt = prompts.get("whole_object", None)
    wick_prompt = prompts.get("wick", None)

    if whole_prompt is None or wick_prompt is None:
        print("    [Error] candle prompts must include 'whole_object' and 'wick'")
        return

    mask_all, _ = engine.predict_mask(whole_prompt)
    if mask_all is None:
        print("    [Error] No candles detected.")
        return
    mask_all = mask_all.astype(bool)

    # 保存集合图：四个蜡烛都在
    engine.save_masked_cutout(mask_all, os.path.join(save_dir, "whole_object.png"))
    print("    [Saved] whole_object.png (All candles)")

    # --- 2) 从集合里选一个蜡烛（默认：面积最大的那个） ---
    mask_uint8 = (mask_all.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("    [Error] No contours found in candle mask.")
        return

    best_contour = max(contours, key=cv2.contourArea)
    mask_single = np.zeros_like(mask_all, dtype=bool)
    cv2.drawContours(mask_single.view(np.uint8), [best_contour], -1, 1, thickness=cv2.FILLED)

    # --- 3) 预测 wick mask，并限制在“单蜡烛”内 ---
    mask_wick, _ = engine.predict_mask(wick_prompt)
    if mask_wick is None:
        print("    [Error] Wick not detected.")
        return
    mask_wick = mask_wick.astype(bool)

    # 关键：只保留单蜡烛内部的 wick，避免背景噪点
    mask_wick = mask_wick & mask_single

    # 再做一次连通域：取最大 wick 组件，去掉散点
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_wick.astype(np.uint8), connectivity=8)
    if num_labels < 2:
        print("    [Warning] No wick component inside selected candle.")
        return

    best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask_wick_single = (labels == best_label)

    # 保存 wick
    engine.save_masked_cutout(mask_wick_single, os.path.join(save_dir, "part_wick.png"))
    print("    [Saved] part_wick.png (One candle wick)")

    # --- 4) wax_surface = 单蜡烛 - wick ---
    mask_wax = mask_single & (~mask_wick_single)

    engine.save_masked_cutout(mask_wax, os.path.join(save_dir, "wax_surface.png"))
    print("    [Saved] wax_surface.png (One candle wax surface)")


def process_fryum(engine, img_name, save_dir, prompts, full_img_path):
    process_hollow_object(engine, img_name, save_dir, prompts["target_object"])

def process_macaroni1(engine, img_name, save_dir, prompts, full_img_path):
    """
    macaroni1: 同时保存
      1) whole_object.png      -> 多个 macaroni 的集合(mask_all)
      2) single_object.png     -> 单个最大连通域(保孔)
    """
    print(f"  [Logic] Processing macaroni1 (Collection + Single, Keep Holes): {img_name}")

    # 1) 预测所有 macaroni 的 mask（多实例）
    mask_all, _ = engine.predict_mask(prompts["target_object"])
    if mask_all is None:
        print("    [Error] No macaroni detected.")
        return

    # 2) 保存“多个”的整体图（集合）
    engine.save_masked_cutout(mask_all.astype(bool), os.path.join(save_dir, "whole_object.png"))
    print("    [Saved] whole_object.png (Collection)")

    # 3) 连通域：取最大连通块作为“单个”实例（保留孔洞）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_all.astype(np.uint8), connectivity=8
    )
    if num_labels < 2:
        print("    [Warning] Only background found in mask.")
        return

    best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask_single = (labels == best_label).astype(bool)

    # 4) 保存“单个”的图
    engine.save_masked_cutout(mask_single, os.path.join(save_dir, "orange elbow macaroni.png"))
    print("    [Saved] orange elbow macaroni.png (Single)")

def process_macaroni2(engine, img_name, save_dir, prompts, full_img_path):
    """
    macaroni2: 同时保存
      1) whole_object.png      -> 多个 macaroni 的集合(mask_all)
      2) single_object.png     -> 单个最大连通域(保孔)
    """
    print(f"  [Logic] Processing macaroni2 (Collection + Single, Keep Holes): {img_name}")

    # 1) 预测所有 macaroni 的 mask（多实例）
    mask_all, _ = engine.predict_mask(prompts["target_object"])
    if mask_all is None:
        print("    [Error] No macaroni detected.")
        return

    # 2) 保存“多个”的整体图（集合）
    engine.save_masked_cutout(mask_all.astype(bool), os.path.join(save_dir, "whole_object.png"))
    print("    [Saved] whole_object.png (Collection)")

    # 3) 连通域：取最大连通块作为“单个”实例（保留孔洞）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_all.astype(np.uint8), connectivity=8
    )
    if num_labels < 2:
        print("    [Warning] Only background found in mask.")
        return

    best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask_single = (labels == best_label).astype(bool)

    # 4) 保存“单个”的图
    engine.save_masked_cutout(mask_single, os.path.join(save_dir, "yellow elbow macaroni.png"))
    print("    [Saved] yellow elbow macaroni.png (Single)")

def process_pipe_fryum(engine, img_name, save_dir, prompts, full_img_path):
    process_hollow_object(engine, img_name, save_dir, prompts["target_object"])

# ================= 核心处理逻辑 =================
def process_chewinggum(engine, img_name, save_dir, prompts, full_img_path):
    # 复用通用单体处理逻辑，只提取 whole_object
    print(f"  [Logic] Processing chewinggum: {img_name}")
    process_single_instance(engine, img_name, save_dir, prompts["target_object"], full_img_path)

# ================= 核心处理逻辑 =================
def process_cashew(engine, img_name, save_dir, prompts, full_img_path):
    # 复用通用单体处理逻辑，只提取 whole_object
    print(f"  [Logic] Processing Cashew: {img_name}")
    process_single_instance(engine, img_name, save_dir, prompts["target_object"], full_img_path)
def process_pcb1(engine, img_name, save_dir, prompts, full_img_path):
    """
    pcb1：直接用 prompt 'cylinders' 切出两个圆柱体
    另一部分 = whole_object - cylinders
    输出：
      - whole_object.png
      - cylinders.png
      - circuit_board.png
    全部背景去除(黑底)，parts 按 bbox 裁剪。
    """
    print(f"  [Logic] Processing PCB1 (Prompt Cylinders + Rest): {img_name}")

    # 1) whole mask：优先 whole_object，其次 whole_board（兼容旧配置）
    whole_prompt = prompts.get("whole_object", None)
    if whole_prompt is None:
        whole_prompt = prompts.get("whole_board", None)
    if whole_prompt is None:
        print("    [Error] Missing prompt: whole_object/whole_board")
        return

    mask_whole, _ = engine.predict_mask(whole_prompt)
    if mask_whole is None:
        print("    [Error] Whole object not detected.")
        return
    mask_whole = mask_whole.astype(bool)

    # 2) cylinders mask：必须提供 prompts["cylinders"]
    cyl_prompt = prompts.get("cylinders", None)
    if cyl_prompt is None:
        print("    [Error] Missing prompt: cylinders (please add prompts['cylinders'])")
        return

    mask_cyl, _ = engine.predict_mask(cyl_prompt)
    if mask_cyl is None:
        print("    [Error] Cylinders not detected.")
        return
    mask_cyl = mask_cyl.astype(bool)

    # 3) 读图
    image = cv2.imread(full_img_path)
    if image is None:
        print(f"    [Error] Could not read image: {full_img_path}")
        return

    # 4) 保存 whole_object（背景去除）
    engine.save_masked_cutout(mask_whole, os.path.join(save_dir, "whole_object.png"))
    print("    [Saved] whole_object.png")

    # 5) 规范化：cylinders 限制在 whole 内，避免切到背景的误检
    mask_cyl = mask_cyl & mask_whole

    # 6) rest = whole - cylinders
    mask_rest = mask_whole & (~mask_cyl)

    # 7) cylinders 输出（黑底+裁剪）
    cyl_img = np.zeros_like(image)
    cyl_img[mask_cyl] = image[mask_cyl]
    bbox_cyl = get_bbox_from_mask(mask_cyl)
    if bbox_cyl:
        x1, y1, x2, y2 = bbox_cyl
        crop_and_save(cyl_img, x1, y1, x2, y2, os.path.join(save_dir, "cylinders.png"))
        print("    [Saved] part_1_cylinders.png")
    else:
        print("    [Warn] No bbox for cylinders.")

    # 8) rest 输出（黑底+裁剪）
    rest_img = np.zeros_like(image)
    rest_img[mask_rest] = image[mask_rest]
    bbox_rest = get_bbox_from_mask(mask_rest)
    if bbox_rest:
        x1, y1, x2, y2 = bbox_rest
        crop_and_save(rest_img, x1, y1, x2, y2, os.path.join(save_dir, "circuit_board.png"))
        print("    [Saved] part_2_board_others.png")
    else:
        print("    [Warn] No bbox for rest.")

    print("    [Success] Saved whole_object + cylinders + rest (background removed).")
def process_pcb2(engine, img_name, save_dir, prompts, full_img_path):
    """
    PCB2 (HC-SR04) 专用逻辑:
    1. 识别整块板子并去背景。
    2. 保存 whole_object.png
    3. 将板子在 X 轴上切分为：左、中、右三部分。
    """
    print(f"  [Logic] Processing PCB2 (Left-Mid-Right Split): {img_name}")

    # 1. 获取整板 Mask
    mask_whole, _ = engine.predict_mask(prompts["whole_board"])
    if mask_whole is None:
        print("    [Error] Board not detected.")
        return

    # 2. 获取 Bounding Box (确定板子的物理边界)
    bbox = get_bbox_from_mask(mask_whole)
    if not bbox:
        return
    x_min, y_min, x_max, y_max = bbox

    board_w = x_max - x_min

    # 3. 读取原图并应用 Mask (关键步骤：去背景)
    image = cv2.imread(full_img_path)
    if image is None:
        return

    # 创建一个纯黑背景的图像，尺寸与原图一致
    masked_img = np.zeros_like(image)

    # 确保 mask 是布尔类型 (True/False)
    mask_bool = mask_whole.astype(bool)

    # 将原图中 mask 为 True (即电路板部分) 的像素拷贝到纯黑图像上
    masked_img[mask_bool] = image[mask_bool]

    # ==========================
    # 【新增】保存 whole_object.png
    # ==========================
    # 方式1：直接用引擎保存（推荐，和你其他类别一致）
    engine.save_masked_cutout(mask_bool, os.path.join(save_dir, "whole_object.png"))
    print("    [Saved] whole_object.png")

    # 4. 定义切割比例 (针对 HC-SR04 调优)
    split_ratio_1 = 0.38  # 左侧结束点
    split_ratio_2 = 0.62  # 右侧开始点

    split_x1 = int(x_min + (board_w * split_ratio_1))
    split_x2 = int(x_min + (board_w * split_ratio_2))

    # 5. 执行切割并保存 (传入的是 masked_img，所以背景是黑的)

    # Part 1: 左侧 (passives)
    crop_and_save(masked_img, x_min, y_min, split_x1, y_max,
                  os.path.join(save_dir, "left_passives.png"))

    # Part 2: 中间 (header pins)
    crop_and_save(masked_img, split_x1, y_min, split_x2, y_max,
                  os.path.join(save_dir, "middle_header_pins.png"))

    # Part 3: 右侧 (controller ic)
    crop_and_save(masked_img, split_x2, y_min, x_max, y_max,
                  os.path.join(save_dir, "right_controller_ic.png"))
def process_pcb3(engine, img_name, save_dir, prompts, full_img_path):
    """
    PCB3 (红外避障模块) 专用逻辑:
    1. 识别整体模块并去背景。
    2. 使用硬编码比例切出中间的蓝色电位器。
    """
    print(f"  [Logic] Processing PCB3 (Geometric Split + Remove BG): {img_name}")

    # 1. 获取整板 Mask
    mask_whole, _ = engine.predict_mask(prompts["whole_object"])
    
    if mask_whole is None:
        print("    [Error] Whole object not detected.")
        return

    # 2. 获取 Bounding Box
    bbox = get_bbox_from_mask(mask_whole)
    if not bbox: return
    x_min, y_min, x_max, y_max = bbox
    
    board_w = x_max - x_min

    # 3. 读取原图并去背景
    image = cv2.imread(full_img_path)
    if image is None: return

    # 创建黑底图
    masked_img = np.zeros_like(image)
    mask_bool = mask_whole.astype(bool)
    masked_img[mask_bool] = image[mask_bool]
    
    engine.save_masked_cutout(mask_bool, os.path.join(save_dir, "whole_object.png"))
    print("    [Saved] whole_object.png")

    # 4. 定义切割比例 (针对 PCB3 蓝色电位器位置估算)
    # 整个模块包含：[引脚]--[芯片区]--[蓝色电位器]--[电阻区]--[红外灯]
    # 估算比例：
    # 左侧结束点 (Start of Blue Pot): 约 40%
    # 右侧开始点 (End of Blue Pot): 约 58%
    split_ratio_1 = 0.385 
    split_ratio_2 = 0.555

    split_x1 = int(x_min + (board_w * split_ratio_1))
    split_x2 = int(x_min + (board_w * split_ratio_2))

    # 5. 执行切割并保存 (使用 masked_img)
    
    # Part 1: 左侧 (Pins + LM393 Chip)
    crop_and_save(masked_img, x_min, y_min, split_x1, y_max, 
                  os.path.join(save_dir, "left_ir_emit_receive.png"))
    
    # Part 2: 中间 (Blue Potentiometer)
    crop_and_save(masked_img, split_x1, y_min, split_x2, y_max, 
                  os.path.join(save_dir, "middle_comparator_pot.png"))

    # Part 3: 右侧 (Resistors + IR LEDs)
    crop_and_save(masked_img, split_x2, y_min, x_max, y_max, 
                  os.path.join(save_dir, "right_header_indicator.png"))

    print("    [Success] Saved 3 parts based on geometric ratios.")
def process_pcb4(engine, img_name, save_dir, prompts, full_img_path):
    """
    PCB4 专用逻辑 (几何四分法):
    1. SAM 识别整块板子，获取 Bounding Box。
    2. 读取原图。
    3. 按几何比例切割成：左侧、中上、中下、最右侧。
    """
    print(f"  [Logic] Processing PCB4 (4-Part Split): {img_name}")

    # 1. 获取整板 Mask
    mask_whole, _ = engine.predict_mask(prompts["whole_board"])
    
    if mask_whole is None:
        print("    [Error] Board not detected. Skipping.")
        return

    # 2. 获取板子的 Bounding Box
    bbox = get_bbox_from_mask(mask_whole)
    if not bbox:
        print("    [Error] Invalid mask bbox.")
        return
    x_min, y_min, x_max, y_max = bbox
    
    board_w = x_max - x_min
    board_h = y_max - y_min
    
    print(f"    [Info] Board bounds: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")

    # 3. 读取原图
    image = cv2.imread(full_img_path)
    if image is None:
        print(f"    [Error] Could not read image: {full_img_path}")
        return
    
    # 创建黑底图
    masked_img = np.zeros_like(image)
    mask_bool = mask_whole.astype(bool)
    masked_img[mask_bool] = image[mask_bool]
    
    engine.save_masked_cutout(mask_bool, os.path.join(save_dir, "whole_object.png"))
    print("    [Saved] whole_object.png")

    # 4. 定义切割比例
    split_x1_ratio = 0.46 
    split_x1_abs = x_min + (board_w * split_x1_ratio)

    split_x2_ratio = 0.84
    split_x2_abs = x_min + (board_w * split_x2_ratio)
    
    split_y_ratio = 0.44
    split_y_abs = y_min + (board_h * split_y_ratio)

    # 5. 执行切割并保存
    # 区域 1: 左侧 USB
    crop_and_save(image, x_min, y_min, split_x1_abs, y_max, 
                  os.path.join(save_dir, "left_usb.png"))
    
    # 区域 2: 中间上部 电阻
    crop_and_save(image, split_x1_abs, y_min, split_x2_abs, split_y_abs, 
                  os.path.join(save_dir, "mid_top_resistors.png"))

    # 区域 3: 中间下部 IC
    crop_and_save(image, split_x1_abs, split_y_abs, split_x2_abs, y_max, 
                  os.path.join(save_dir, "mid_bottom_ic.png"))

    # 区域 4: 右侧 BAT
    crop_and_save(image, split_x2_abs, y_min, x_max, y_max, 
                  os.path.join(save_dir, "right_bat.png"))

    print("    [Success] Saved 4 geometric parts.")

def process_single_instance(engine, img_name, save_dir, prompt, full_img_path):
    """
    通用逻辑 (Candle/Capsule): 
    1. 检测图中所有目标。
    2. 挑选其中一个(面积最大的/最完整的)。
    3. 只保存一张 whole_object.png (Mask Cutout)。
    """
    print(f"  [Logic] Processing Single Instance (Pick One): {img_name}")

    # 1. 获取 Mask
    mask_all, _ = engine.predict_mask(prompt)
    
    if mask_all is None:
        print("    [Error] No objects detected.")
        return

    # 2. 分离个体 (利用连通域)
    mask_uint8 = (mask_all.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("    [Error] No contours found in mask.")
        return

    # 3. 挑选一个最好的 (选面积最大的，避免选到噪点)
    best_contour = max(contours, key=cv2.contourArea)
    
    # 创建只包含这一个物体的 Mask
    mask_single = np.zeros_like(mask_all, dtype=bool)
    cv2.drawContours(mask_single.view(np.uint8), [best_contour], -1, 1, thickness=cv2.FILLED)

    # 4. 保存
    engine.save_masked_cutout(mask_single, os.path.join(save_dir, "whole_object.png"))
    print("    [Success] Saved whole_object.png")

# 映射函数

def process_capsules(engine, img_name, save_dir, prompts, full_img_path):
    """
    Capsule 新逻辑:
    1. 检测图中"所有"胶囊 -> 保存为 whole_object.png (整体视图)
    2. 从 Mask 中提取一个单体 (面积最大或随机) -> 保存为 single_capsule.png (细粒度视图)
    """
    print(f"  [Logic] Processing Capsule (Whole Collection + Single Sample): {img_name}")

    # --- 步骤 1: 获取所有胶囊 (Whole Object) ---
    mask_all, _ = engine.predict_mask(prompts["target_object"])
    
    if mask_all is None:
        print("    [Error] No capsules detected.")
        return

    # 保存包含所有胶囊的图
    # 注意：这里背景会被 mask 掉，只保留一堆胶囊
    engine.save_masked_cutout(mask_all, os.path.join(save_dir, "whole_object.png"))
    print("    [Saved] whole_object.png (Collection)")

    # --- 步骤 2: 提取单体 (Single Instance) ---
    # 利用连通域分离个体
    mask_uint8 = (mask_all.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("    [Warning] Could not split instances.")
        return

    # 策略：选面积最大的一个 (代表最清晰/完整的样本)
    # 你也可以改为 random.choice(contours) 来“随便取一个”
    best_contour = max(contours, key=cv2.contourArea)
    
    # 创建单体 Mask
    mask_single = np.zeros_like(mask_all, dtype=bool)
    cv2.drawContours(mask_single.view(np.uint8), [best_contour], -1, 1, thickness=cv2.FILLED)

    # 保存单体图
    engine.save_masked_cutout(mask_single, os.path.join(save_dir, "single_capsule.png"))
    print("    [Saved] single_capsule.png (Detail Sample)")
# ================= 主程序 =================

def main():
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser(description="VisA Dataset SAM3 Segmentation Runner")
    parser.add_argument("--category", "-c", type=str, default="capsule", 
                        help="VisA category name (pcb4, candle, capsule)")
    parser.add_argument("--input_dir", type=str, default=BASE_DATA_DIR, help="Base directory")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_ROOT, help="Output root")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT, help="SAM3 checkpoint")

    args = parser.parse_args()
    target_category = args.category.lower()

    # 2. 检查配置
    if target_category not in CATEGORY_CONFIG:
        print(f"Error: Category '{target_category}' is not defined.")
        return

    # 3. 初始化引擎
    try:
        engine = Sam3Engine(args.ckpt)
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    # 4. 路径处理
    input_path = os.path.join(args.input_dir, target_category)
    output_path = os.path.join(args.output_dir, target_category)
    
    print(f"\n=== Starting VisA Task: {target_category} ===")
    
    # 5. 获取图片列表
    cat_conf = CATEGORY_CONFIG[target_category]
    image_list = cat_conf.get("demo_images")
    
    # 自动扫描文件夹兜底
    if not image_list or not os.path.exists(os.path.join(input_path, image_list[0])):
        if os.path.exists(input_path):
            image_list = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
            image_list.sort()
        else:
            print(f"Error: Input path {input_path} not found.")
            return

    # 6. 循环处理
    for img_name in image_list:
        full_img_path = os.path.join(input_path, img_name)
        if not os.path.exists(full_img_path): continue

        engine.set_image(full_img_path)
        
        current_save_dir = os.path.join(output_path, os.path.splitext(img_name)[0])
        if not os.path.exists(current_save_dir):
            os.makedirs(current_save_dir)
        
        func_name = f"process_{target_category}"
        processor_func = globals().get(func_name)
        
        if processor_func:
            processor_func(engine, img_name, current_save_dir, cat_conf["prompts"], full_img_path)
        else:
            print(f"Error: No function {func_name} found.")

    print(f"\nProcessing complete for category: {target_category}")

if __name__ == "__main__":
    main()
