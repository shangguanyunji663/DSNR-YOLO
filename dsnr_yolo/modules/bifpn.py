"""BiFPN 加权特征融合节点（论文 2.4 节的融合单元实现）

BiFPN = Bidirectional Feature Pyramid Network（Tan et al., EfficientDet, CVPR 2020）。
传统 FPN 把不同尺度的特征图直接【相加】或【拼接】，等权对待每一路输入；
BiFPN 的核心改进是给每一路输入学习一个权重 w_i，做"加权求和"：

    快速归一化融合（Fast Normalized Fusion）:
        w_i' = ReLU(w_i) / ( Σ_j ReLU(w_j) + ε )      ← 保证权重非负、和为 1
        out  = SiLU( Σ_i w_i' × x_i )

优点：比 softmax 归一化更快（不需要指数运算），且权重约束在 [0,1] 且和为 1，
训练稳定；在背景占比极高的遥感图像里，模型可以自动学到"更关注小目标所在的
浅层高分辨率特征"（论文 2.4 节末尾）。

使用方式
--------
在 yaml 里写成多输入节点，例如：
    [[12, 14], 1, BiFPNFusion, []]   # 融合第 12 层和第 14 层（要求两路通道数一致！）
parse_model 会自动把 (通道数, 输入路数) 传进来。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPNFusion(nn.Module):
    """BiFPN 的加权融合节点（可学习逐路权重）。

    Parameters
    ----------
    c : int   融合的通道数（要求所有输入特征图通道数都等于 c）
    n : int   输入路数（由 parse_model 根据 yaml 自动传入）
    eps : float  防止除零的小常数
    """

    def __init__(self, c: int, n: int = 2, eps: float = 1e-4):
        super().__init__()
        self.n = n
        self.c = c
        self.eps = eps
        self.w = nn.Parameter(torch.ones(n))   # 每路一个可学习权重，初始化为 1
        self.act = nn.SiLU()

    def forward(self, x):
        # ultralytics 会把多输入整理成 list 传入；单输入时包一层
        if not isinstance(x, list):
            x = [x]
        assert len(x) == self.n, (
            f"BiFPNFusion 期望 {self.n} 路输入，实际收到 {len(x)} 路。"
            "请检查 yaml 中该层的 from 列表长度。"
        )
        for xi in x:
            assert xi.shape[1] == self.c, (
                f"BiFPNFusion 期望通道数 {self.c}，收到 {tuple(xi.shape)}。"
                "请用 1×1 Conv 先对齐各路通道数。"
            )
        # 快速归一化：ReLU 保证非负，除以总和保证和为 1
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = w[0] * x[0]
        for wi, xi in zip(w[1:], x[1:]):
            out = out + wi * xi
        return self.act(out)

    def extra_repr(self) -> str:
        return f"c={self.c}, n_inputs={self.n}"
