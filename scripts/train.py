"""DSNR-YOLO 训练脚本（超参数与论文 3.1.1 节一致）

论文实验设置：
    GPU: NVIDIA GeForce RTX 2070S (8GB)          本机: RTX 4060 Laptop (8GB) ✓
    Python 3.10.16 + PyTorch 2.0.0 + CUDA 11.8   本机: ML2 环境自带版本
    batch=8, epochs=200
    优化器 SGD, lr=0.01, weight_decay=0.0005, momentum=0.9

用法示例：
    # 玩具数据集冒烟（1 分钟验证流程通了没有；nbs=batch 保证真的会更新权重）
    python scripts/train.py --data data/toy.yaml --epochs 3 --imgsz 320 --batch 4 --nbs 4

    # 正式复现（需先按 docs/09 准备好 DIOR 数据集）
    python scripts/train.py --data data/dior.yaml --epochs 200 --imgsz 640 --batch 8 --device 0
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsnr_yolo import register  # noqa: E402  必须在 import YOLO 之前打补丁

register.apply()


def main():
    ap = argparse.ArgumentParser(description="训练 DSNR-YOLO")
    ap.add_argument("--data", default=str(ROOT / "data" / "toy.yaml"), help="数据集 yaml 路径")
    ap.add_argument("--epochs", type=int, default=200, help="训练轮数（论文: 200）")
    ap.add_argument("--imgsz", type=int, default=640, help="输入尺寸（论文按 GFLOPs 推断为 640）")
    ap.add_argument("--batch", type=int, default=8, help="批大小（论文: 8）")
    ap.add_argument("--device", default=None, help="'0' 用第一块GPU, 'cpu' 用CPU, 默认自动")
    ap.add_argument("--weights", default=None, help="从 .pt 恢复/微调（可选）")
    ap.add_argument("--nbs", type=int, default=None, help="名义批大小(梯度累积)。小数据集冒烟时设为与 --batch 相同，"
                                                     "否则累积太多次会导致从不到 optimizer.step")
    ap.add_argument("--name", default="dsnr_yolo", help="本次运行的目录名")
    args = ap.parse_args()

    from ultralytics import YOLO

    # 自动选设备
    import torch

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    # 构建模型：有权重用权重，否则从 yaml 从零构建
    if args.weights:
        model = YOLO(args.weights)          # .pt 里已含结构，类路径可导入（register 已生效）
        if not register.is_dsnr(model):
            print("[提示] 加载的不是 DSNR-YOLO 权重，结构以权重文件为准")
    else:
        model = YOLO(str(ROOT / "dsnr_yolo" / "models" / "dsnr_yolo.yaml"), task="detect")
        register.swap_class(model)          # 换成带混合损失的模型类（docs/08 详解）

    model.info(verbose=True)

    extra = {}
    if args.nbs:
        extra["nbs"] = args.nbs   # 梯度累积的名义批大小（默认 64，小数据集需调小）

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        optimizer="SGD",          # 论文: 随机梯度下降
        lr0=0.01,                 # 论文: 初始学习率 0.01
        momentum=0.9,             # 论文: 动量 0.9
        weight_decay=0.0005,      # 论文: 权重衰减 0.0005
        workers=2,                # Windows 下线程过多易报错，保守取 2
        project=str(ROOT / "runs"),
        name=args.name,
        seed=0,                   # 固定随机种子，便于复现对比
        exist_ok=True,
        **extra,
    )


if __name__ == "__main__":
    main()
