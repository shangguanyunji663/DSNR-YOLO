"""DSNR-YOLO 验证/评估脚本

在验证集或测试集上计算 mAP50、mAP50-95、Precision、Recall 等指标。

用法示例：
    python scripts/val.py --data data/toy.yaml --weights runs/dsnr_yolo/weights/best.pt
    python scripts/val.py --data data/dior.yaml --weights runs/dsnr_yolo/weights/best.pt --device 0
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsnr_yolo import register  # noqa: E402

register.apply()


def main():
    ap = argparse.ArgumentParser(description="验证 DSNR-YOLO")
    ap.add_argument("--data", default=str(ROOT / "data" / "toy.yaml"), help="数据集 yaml")
    ap.add_argument("--weights", default=str(ROOT / "runs" / "dsnr_yolo" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="val", choices=["val", "test"], help="评估哪个子集")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        device=args.device,
        project=str(ROOT / "runs"),
        name="val_dsnr_yolo",
        exist_ok=True,
    )

    # 摘要与论文表 5 的指标口径对应
    print("\n================ 评估结果 ================")
    print(f"mAP50    : {metrics.box.map50 * 100:.1f} %   (论文 DIOR: 89.0)")
    print(f"mAP50-95 : {metrics.box.map * 100:.1f} %   (论文 DIOR: 66.9)")
    print(f"Precision: {metrics.box.mp * 100:.1f} %")
    print(f"Recall   : {metrics.box.mr * 100:.1f} %")


if __name__ == "__main__":
    main()
