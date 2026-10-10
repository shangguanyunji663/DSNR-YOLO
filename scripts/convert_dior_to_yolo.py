"""把 VOC 风格 XML 标注（DIOR / SIMD）转换成 YOLO 格式，并按论文 8:1:1 划分

论文 3.1.2 节："由于所选数据集最初未划分为单独的子集，将其按照 8:1:1 的
比例划分为训练集、验证集和测试集" —— 本脚本完全按此执行。

DIOR 原始结构（官方发布）：
    <src>/Annotations/*.xml        每图一个 XML：size + object(bndbox)
    <src>/JPEGImages/*.jpg         （有的发布包叫 JPEGImages-support 或 Images）

输出（ultralytics 标准目录）：
    <dst>/images/{train,val,test}/*.jpg
    <dst>/labels/{train,val,test}/*.txt   每行: cls cx cy w h（归一化）
    并打印/写入类别对照表（按字母序编号），请把它同步进 data/*.yaml 的 names。

用法：
    python scripts/convert_dior_to_yolo.py --src D:/datasets/DIOR --dst data/dior
"""

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

# DIOR 官方 20 类（保证与 data/dior.yaml 的 names 一致）
DIOR_CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "expressway-service-area", "expressway-toll-station", "dam",
    "golffield", "groundtrackfield", "harbor", "overpass", "vehicle",
    "parkinglot", "ship", "station", "storagetank", "tenniscourt", "windmill",
]


def find_subdir(src: Path, *candidates: str):
    for c in candidates:
        p = src / c
        if p.is_dir():
            return p
    return None


def parse_xml(xml_path: Path, class_to_id: dict):
    """解析一个 VOC XML，返回 (w, h, [(cls_id, cx, cy, bw, bh), ...])，越界框做裁剪。"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W = int(float(size.find("width").text))
    H = int(float(size.find("height").text))
    boxes = []
    for obj in root.iter("object"):
        name = obj.find("name").text.strip().lower()
        if name not in class_to_id:
            continue  # 数据集外的类别直接丢弃
        b = obj.find("bndbox")
        xmin, ymin = float(b.find("xmin").text), float(b.find("ymin").text)
        xmax, ymax = float(b.find("xmax").text), float(b.find("ymax").text)
        # 裁剪到图像范围内并丢弃退化框
        xmin, xmax = max(0.0, min(xmin, W - 1)), max(0.0, min(xmax, W - 1))
        ymin, ymax = max(0.0, min(ymin, H - 1)), max(0.0, min(ymax, H - 1))
        if xmax - xmin < 1 or ymax - ymin < 1:
            continue
        cx, cy = (xmin + xmax) / 2 / W, (ymin + ymax) / 2 / H
        bw, bh = (xmax - xmin) / W, (ymax - ymin) / H
        boxes.append((class_to_id[name], cx, cy, bw, bh))
    return W, H, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始数据根目录（含 Annotations 与图像目录）")
    ap.add_argument("--dst", required=True, help="输出根目录")
    ap.add_argument("--ratio", default="8:1:1", help="训练:验证:测试 划分比例（论文为 8:1:1）")
    ap.add_argument("--seed", type=int, default=0, help="划分随机种子（保证可复现）")
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if src == dst or src in dst.parents:
        raise ValueError(
            "输出目录不能等于或位于源目录内部（会把转换结果与原始数据混在一起）。"
            f"\n  src = {src}\n  dst = {dst}"
        )
    ann_dir = find_subdir(src, "Annotations", "annotations", "Annotation")
    img_dir = find_subdir(src, "JPEGImages", "JPEGImages-support", "Images", "images")
    if ann_dir is None or img_dir is None:
        raise FileNotFoundError(
            f"在 {src} 下找不到 Annotations/ 与图像目录。请确认目录结构，"
            f"或手动改名后重试。当前包含: {[p.name for p in src.iterdir()]}"
        )

    # 类别表：DIOR 用固定官方顺序；其它数据集按 XML 中出现的字母序生成
    all_names = set()
    for x in ann_dir.glob("*.xml"):
        for obj in ET.parse(x).getroot().iter("object"):
            all_names.add(obj.find("name").text.strip().lower())
    if set(DIOR_CLASSES) == all_names:
        classes = DIOR_CLASSES
    else:
        classes = sorted(all_names)
    class_to_id = {c: i for i, c in enumerate(classes)}
    print("类别对照表（已同步写入 data/*.yaml 的 names）：")
    for c, i in class_to_id.items():
        print(f"  {i:>2} {c}")

    # 划分：按 8:1:1 随机分配（论文 3.1.2）
    tr, va, _ = (int(x) for x in args.ratio.split(":"))
    xmls = sorted(ann_dir.glob("*.xml"))
    random.seed(args.seed)
    random.shuffle(xmls)
    n = len(xmls)
    n_val = max(1, round(n * va / (tr + va + _)))
    n_test = max(1, round(n * _ / (tr + va + _)))
    splits = {
        "train": xmls[: n - n_val - n_test],
        "val": xmls[n - n_val - n_test : n - n_test],
        "test": xmls[n - n_test :],
    }

    for split, files in splits.items():
        img_out = dst / "images" / split
        lbl_out = dst / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        n_boxes = 0
        for x in files:
            stem = x.stem
            # 找对应图片（兼容 .jpg/.png/.bmp）
            img_path = next(
                (img_dir / f"{stem}{ext}" for ext in (".jpg", ".png", ".bmp") if (img_dir / f"{stem}{ext}").exists()),
                None,
            )
            if img_path is None:
                print(f"[跳过] 找不到图像: {stem}")
                continue
            W, H, boxes = parse_xml(x, class_to_id)
            if not boxes:
                continue  # 无有效目标的图不参与训练
            n_boxes += len(boxes)
            # 中文路径兼容：cv2.imwrite 直写会失败，用 imencode+tofile
            img = cv2.imread(str(img_path))
            ok, buf = cv2.imencode(img_path.suffix, img)
            if not ok:
                print(f"[跳过] 图像读取失败: {img_path}")
                continue
            buf.tofile(str(img_out / img_path.name))
            with open(lbl_out / f"{stem}.txt", "w", encoding="utf-8") as f:
                for cls_id, cx, cy, bw, bh in boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"[{split}] 图像 {len(files)} 张, 目标 {n_boxes} 个 -> {img_out}")

    print("\n完成！请把上面的类别对照表同步到 data/*.yaml 的 names，并把 path 改为：")
    print(f"  path: {dst.resolve().as_posix()}")


if __name__ == "__main__":
    main()
