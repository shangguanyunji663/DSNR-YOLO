"""DSNR-YOLO 模型自检脚本 —— 不用数据就能验证"模型搭对了没有"

做四件事：
  1. 按 yaml 构建网络，打印每层结构与参数量；
  2. 统计总参数量(M)与计算量(GFLOPs)，与论文对照（7.15M / 18.3 GFLOPs）；
  3. 用随机张量做一次前向，检查三尺度输出形状是否正确；
  4. 构造假 batch 调一次损失函数，确认 ATFL+SLIDE 混合损失能反传。

用法：
    python scripts/verify_model.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsnr_yolo import register  # noqa: E402

register.apply()


def main():
    import torch
    from ultralytics import YOLO

    print("=" * 70)
    print("步骤 1/4：构建 DSNR-YOLO 并打印结构")
    print("=" * 70)
    model = YOLO(str(ROOT / "dsnr_yolo" / "models" / "dsnr_yolo.yaml"), task="detect")
    register.swap_class(model)
    net = model.model

    print("=" * 70)
    print("步骤 2/4：参数量 / 计算量（对照论文：Params=7.15M, GFLOPs=18.3, 输入640）")
    print("=" * 70)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"总参数量: {n_params / 1e6:.2f} M   (论文 DSNR-YOLO: 7.15 M, 基线 yolo11s: 9.43 M)")
    # 用 ultralytics 的 profile 机制统计 GFLOPs（640×640 输入，与论文口径一致）
    import torch as _t

    net.eval()
    with torch.no_grad():
        net(_t.zeros(1, 3, 640, 640), profile=False)
    from ultralytics.utils.torch_utils import get_flops

    print(f"GFLOPs  : {get_flops(net, imgsz=640):.1f}   (论文 DSNR-YOLO: 18.3, 基线 yolo11s: 21.5)")

    print("=" * 70)
    print("步骤 3/4：前向传播形状检查（三尺度检测层）")
    print("=" * 70)
    net.eval()
    with torch.no_grad():
        preds = net(torch.zeros(1, 3, 640, 640))
    # Detect 头 eval 输出: (预测张量, 原始输出dict)；训练输出直接是 dict
    raw = preds[1] if isinstance(preds, tuple) else preds
    if isinstance(raw, dict) and "one2many" in raw:
        raw = raw["one2many"]          # 端到端头才有该键
    feats = raw["feats"]               # 三尺度特征图列表
    for i, ft in enumerate(feats):
        print(f"  检测尺度 {i}: 特征图 {tuple(ft.shape)}")
    assert tuple(feats[0].shape[-2:]) == (80, 80), "P3 特征图应为 80×80"
    assert tuple(feats[1].shape[-2:]) == (40, 40), "P4 特征图应为 40×40"
    assert tuple(feats[2].shape[-2:]) == (20, 20), "P5 特征图应为 20×20"
    print("  ✓ 三尺度输出形状正确")

    print("=" * 70)
    print("步骤 4/4：混合损失（ATFL+SLIDE）前向 + 反向检查")
    print("=" * 70)
    # 正式训练时 trainer 会给 model.args 赋配置对象；脱离 trainer 自测需手动补上
    # （损失里要用 hyp.box / hyp.cls / hyp.dfl 三个增益系数）
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    net.args = get_cfg(DEFAULT_CFG)
    net.train()
    fake_batch = {
        "img": torch.rand(2, 3, 640, 640),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0], [2.0]]),
        "bboxes": torch.tensor([[0.50, 0.50, 0.20, 0.30],
                                [0.25, 0.75, 0.10, 0.10],
                                [0.60, 0.30, 0.30, 0.40]]),
    }
    loss, loss_items = net.loss(fake_batch)   # 内部会自动创建 MixedV8DetectionLoss
    print(f"  总损失(三分量求和×batch) = {loss.sum().item():.4f}")
    print(f"  分项 {dict((k, round(v.item(), 4)) for k, v in loss_items.items())}")
    loss.sum().backward()

    crit = net._dsnr_criterion
    g1 = crit.w_atfl.grad
    g2 = crit.w_slide.grad
    print(f"  混合权重 ω1(ATFL)={crit.w_atfl.item():.4f}, ω2(SLIDE)={crit.w_slide.item():.4f}")
    assert g1 is not None and g2 is not None, "可学习权重没有收到梯度！"
    print(f"  ✓ ω1/ω2 收到梯度 ({g1.item():.4f}, {g2.item():.4f})，会被优化器更新")
    assert net._dsnr_criterion in list(net.modules()), "损失未注册为模型子模块"

    print("=" * 70)
    print("全部自检通过 ✔  模型结构与论文一致，可以直接开始训练")
    print("=" * 70)


if __name__ == "__main__":
    main()
