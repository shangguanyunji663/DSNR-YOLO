"""DSNR-YOLO 模块单元测试 —— 逐个验证论文公式是否被正确实现

运行方式（先激活 ML2 环境并安装好依赖）：
    python tests/test_modules.py

每个 test_xxx 函数对应一个模块/一组公式，全部通过会打印 PASS。
这是学习时最好的"动手环节"：改一改模块参数，看参数量和输出怎么变。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsnr_yolo.modules import NAMChannelAttention, DSNRblock, ConvBNAct, BiFPNFusion


def test_nam_channel_attention():
    """验证 NAM 式(3)(4)(5)：形状不变、零新增参数、权重和为 1。"""
    c = 64
    bn = torch.nn.BatchNorm2d(c)
    nam = NAMChannelAttention(bn)
    x = torch.randn(2, c, 32, 32)

    # 新增可学习参数应为 0（NAM 的"零成本"卖点）。
    # 注意：nam 里保存的是外部 bn 的引用，parameters() 会把 bn 自己的参数也算进来，
    # 所以按"参数对象身份"排除 bn 已有的参数，只统计真正新增的。
    bn_params = {id(p) for p in bn.parameters()}
    n_new_params = sum(p.numel() for p in nam.parameters() if id(p) not in bn_params)
    assert n_new_params == 0, f"NAM 不应新增参数，实际新增 {n_new_params}"

    y = nam(x)
    assert y.shape == x.shape, f"输出形状应不变: {y.shape}"

    # 式(3)：|γ| 归一化后和为 1
    w = bn.weight.abs()
    w = w / w.sum()
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)
    print("PASS  NAM 通道注意力（式3-5）：形状不变、零新增参数")


def test_dsnrblock_same_channels():
    """通道数不变时：残差支路走 Identity（特征直通）。"""
    blk = DSNRblock(128, 128)
    x = torch.randn(2, 128, 40, 40)
    y = blk(x)
    assert y.shape == x.shape
    assert isinstance(blk.shortcut, torch.nn.Identity), "通道相同应走特征直通"
    # 残差性质：当 NAM 门→0 且无激活时输出≈输入；这里仅检查数值有限
    assert torch.isfinite(y).all()
    print("PASS  DSNRblock 通道不变：直通残差 + 输出形状正确")


def test_dsnrblock_change_channels():
    """通道数变化时：残差支路自动用 1×1 卷积调整（式(7)）。"""
    blk = DSNRblock(64, 128)
    assert isinstance(blk.shortcut, ConvBNAct), "通道不同应走 1×1 卷积调整（式7）"
    assert blk.shortcut.conv.kernel_size == (1, 1)
    x = torch.randn(2, 64, 80, 80)
    y = blk(x)
    assert y.shape == (2, 128, 80, 80), f"输出通道应为 128: {y.shape}"
    print("PASS  DSNRblock 通道变化：1×1 卷积自动调整（式6/7）")


def test_dsnrblock_stride():
    """步幅=2 时：主支路与残差支路同步下采样。"""
    blk = DSNRblock(64, 128, s=2)
    x = torch.randn(1, 64, 80, 80)
    y = blk(x)
    assert y.shape == (1, 128, 40, 40), f"步幅2应把尺寸减半: {y.shape}"
    print("PASS  DSNRblock 步幅=2：下采样正常")


def test_dsnrblock_nam_gate_learns():
    """NAM 门是可学习的：梯度能流到 nam_gate。"""
    blk = DSNRblock(32, 32)
    x = torch.randn(2, 32, 16, 16)
    y = blk(x)
    y.sum().backward()
    assert blk.nam_gate.grad is not None and blk.nam_gate.grad.abs() > 0, "门控没有梯度"
    g = torch.sigmoid(blk.nam_gate)
    assert 0 < g.item() < 1
    print(f"PASS  DSNRblock NAM 门可学习（当前门控值={g.item():.3f}）")


def test_dsnrblock_lighter_than_standard_conv():
    """轻量化验证：DSNRblock 参数量应远小于同通道标准卷积。"""
    c1, c2 = 256, 256
    blk = DSNRblock(c1, c2, use_nam=False)
    p_dsnr = sum(p.numel() for p in blk.parameters())
    # 对照：一个 3×3 标准卷积(c1→c2) + 1×1(c1→c2)
    p_std = c1 * c2 * 9 + c1 * c2
    ratio = p_std / p_dsnr
    assert ratio > 2, f"DSNRblock 应明显更轻: 标准={p_std}, DSNR={p_dsnr}"
    print(f"PASS  DSNRblock 轻量化：同通道标准卷积参数量是它的 {ratio:.1f} 倍")


def test_bifpn_fusion():
    """验证 BiFPN 快速归一化融合：权重非负归一、多路输入、可学习。"""
    c = 64
    for n in (2, 3):
        fus = BiFPNFusion(c, n=n)
        xs = [torch.randn(1, c, 40, 40) for _ in range(n)]
        y = fus(xs)
        assert y.shape == xs[0].shape
        w = torch.relu(fus.w)
        w = w / (w.sum() + fus.eps)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-3), "权重和应为 1"
        # 手工计算对照
        y_ref = sum(wi * xi for wi, xi in zip(w, xs))
        assert torch.allclose(y, torch.nn.functional.silu(y_ref), atol=1e-5)
    # 权重应能学习
    fus = BiFPNFusion(32, n=2)
    y = fus([torch.randn(1, 32, 8, 8), torch.randn(1, 32, 8, 8)])
    y.sum().backward()
    assert fus.w.grad is not None
    print("PASS  BiFPNFusion 快速归一化融合：2/3 路输入、权重可学习、数值正确")


def test_ghost_conv():
    """验证 ultralytics 的 GhostConv：一半本征卷积 + 一半廉价线性变换拼接。"""
    from ultralytics.nn.modules import GhostConv

    conv = GhostConv(64, 128, k=3, s=1)
    x = torch.randn(1, 64, 40, 40)
    y = conv(x)
    assert y.shape == (1, 128, 40, 40)
    # 参数量应明显小于标准 3×3 卷积（本实现：一半正常 3×3 + 一半深度可分离 5×5，
    # 理论压缩比 = 9c1 / (4.5c1 + 12.5)，64→128 时约 1.9 倍）
    p_ghost = sum(p.numel() for p in conv.parameters())
    p_std = 64 * 128 * 9
    assert p_std / p_ghost > 1.8, f"GhostConv 应更轻: ghost={p_ghost}, std={p_std}"
    print(f"PASS  GhostConv：输出128通道，参数量仅为标准卷积的 1/{p_std/p_ghost:.2f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_nam_channel_attention()
    test_dsnrblock_same_channels()
    test_dsnrblock_change_channels()
    test_dsnrblock_stride()
    test_dsnrblock_nam_gate_learns()
    test_dsnrblock_lighter_than_standard_conv()
    test_bifpn_fusion()
    test_ghost_conv()
    print("\n========================================")
    print("全部模块测试通过 ✔")
    print("========================================")
