"""生成微型合成数据集（toy dataset）—— 让你在没有真实遥感数据时也能跑通全流程

原理：用 OpenCV 画随机颜色/位置/大小的 三种图形（矩形块、圆、三角形）模拟
"遥感图像上的目标"，并按 YOLO 格式生成标签。图像模拟"卫星俯视视角"：
深色底 + 噪声 + 小图形。

生成的目录结构（ultralytics 标准格式）：
    data/toy/
        images/train/*.jpg   images/val/*.jpg
        labels/train/*.txt   labels/val/*.txt
    data/toy.yaml

标签 txt 每行: class cx cy w h（都是 0~1 的归一化坐标）
"""

import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = 320          # 小图训练快，CPU 也能跑
N_TRAIN, N_VAL = 40, 8
CLASSES = ["block", "circle", "triangle"]


def _safe_path(*parts) -> Path:
    """把相对路径解析为项目根目录下的绝对路径，并校验没有越出项目目录。

    说明：本脚本所有路径都由固定常量拼接、不接收任何用户输入，
    这里仍做 resolve + 前缀校验，作为防御性编程的示范。
    """
    p = (ROOT.joinpath(*parts)).resolve()
    root = ROOT.resolve()
    if root != p and root not in p.parents:
        raise ValueError(f"路径越出项目目录: {p}")
    return p


def draw_one(img: np.ndarray, cls_id: int) -> tuple:
    """在图上画一个目标，返回 YOLO 标签 (cls, cx, cy, w, h)，均为归一化值。

    注意：这里刻意用固定种子的 random 而非加密随机源——
    目的是让每次生成的数据集完全一致（教学可复现），随机性不做安全用途。
    """
    H, W = img.shape[:2]
    size = random.randint(18, 48)                       # 目标边长/直径（像素）
    x0 = random.randint(0, W - size - 1)
    y0 = random.randint(0, H - size - 1)
    color = tuple(int(random.randint(60, 255)) for _ in range(3))

    if cls_id == 0:      # 矩形块（模拟建筑/车辆顶视）
        cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), color, -1)
    elif cls_id == 1:    # 圆（模拟储罐）
        cv2.circle(img, (x0 + size // 2, y0 + size // 2), size // 2, color, -1)
    else:                # 三角形（模拟屋顶）
        pts = np.array([[x0 + size // 2, y0],
                        [x0, y0 + size],
                        [x0 + size, y0 + size]], np.int32)
        cv2.fillPoly(img, [pts], color)

    cx, cy = (x0 + size / 2) / W, (y0 + size / 2) / H
    return cls_id, cx, cy, size / W, size / H


def make_split(split: str, n_imgs: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    img_dir = _safe_path("data", "toy", "images", split)
    lbl_dir = _safe_path("data", "toy", "labels", split)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for k in range(n_imgs):
        # 深色"地表"底图 + 随机噪声，模拟遥感背景
        img = np.random.randint(20, 60, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        labels = []
        for _ in range(random.randint(1, 4)):           # 每张图 1~4 个目标
            cls_id = random.randint(0, len(CLASSES) - 1)
            labels.append(draw_one(img, cls_id))
        # 轻微高斯模糊，模拟遥感成像
        img = cv2.GaussianBlur(img, (3, 3), 0)
        name = f"{split}_{k:04d}"
        # 注意：Windows 下 cv2.imwrite 不支持含中文的路径，必须用 imencode+tofile
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError(f"图像编码失败: {name}")
        buf.tofile(str(img_dir / f"{name}.jpg"))
        with open(lbl_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            for cls_id, cx, cy, w, h in labels:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    print(f"[toy] {split}: 生成 {n_imgs} 张图 -> {img_dir}")


def main():
    make_split("train", N_TRAIN, seed=0)
    make_split("val", N_VAL, seed=100)

    yaml_path = _safe_path("data", "toy.yaml")
    yaml_path.write_text(
        f"# 自动生成的玩具数据集配置（仅用于跑通流程，不代表真实精度）\n"
        f"path: {_safe_path('data', 'toy').as_posix()}\n"
        f"train: images/train\nval: images/val\nnc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )
    print(f"[toy] 数据集配置 -> {yaml_path}")


if __name__ == "__main__":
    main()
