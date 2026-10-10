"""NAM 通道注意力模块（论文式(3)–(5) 的 1:1 实现）

NAM = Normalization-based Attention Module（Liu et al., 2021, arXiv:2111.12419）。
DSNR-YOLO 论文只使用了 NAM 的【通道注意力】分支，核心思想一句话：

    "通道重不重要，看 BatchNorm 的缩放参数 γ 就知道"

BatchNorm 的 γ（即 bn.weight）在训练中本来就在学习"每个通道该被放大还是缩小"，
NAM 直接拿它当注意力权重 —— 不引入任何新参数，这是它轻量的关键。

论文公式对照
------------
式(3):  weight = |bn.weight_i| / Σ_i |bn.weight_i|     —— 把 |γ| 归一化成和为 1 的权重
式(4):  x_att  = weight ⊙ x                            —— 逐通道加权
式(5):  x_out  = sigmoid(x_att) ⊙ x                    —— Sigmoid 门控后再次加权

为什么适合遥感小目标（论文 2.2 节）：
  小目标特征稀疏、易被背景噪声掩盖、有效通道占比低。
  - BN 的归一化本身能消除背景噪声带来的全局强度偏移（"归一化降噪"）；
  - 通道注意力会给"表征目标关键信息"的通道分配高权重（"通道强化"）。
"""

import torch
import torch.nn as nn


class NAMChannelAttention(nn.Module):
    """基于 BatchNorm 权重的通道注意力（零新增参数）。

    Parameters
    ----------
    bn : nn.BatchNorm2d
        前面卷积层已有的 BN 模块。注意：我们【复用】它的 weight(γ)，
        而不是新建一个 BN —— 这正是 NAM "基于归一化" 的含义。
    eps : float
        防止除零的小常数。
    """

    def __init__(self, bn: nn.BatchNorm2d, eps: float = 1e-12):
        super().__init__()
        self.bn = bn
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        w = self.bn.weight.abs()                    # |γ|, 形状 (C,)
        w = w / (w.sum() + self.eps)                # 式(3)：归一化，和为 1
        x_att = w.view(1, -1, 1, 1) * x             # 式(4)：逐通道缩放 (广播到 H, W)
        return torch.sigmoid(x_att) * x             # 式(5)：门控加权，输出形状不变

    def extra_repr(self) -> str:
        return f"channels={self.bn.num_features} (复用外部 BN 的 γ, 无新增参数)"
