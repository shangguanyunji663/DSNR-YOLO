"""DSNRblock —— 论文的核心模块（2.2 节 + 图 2 的代码复现）

名字含义：Depthwise Separable convolution with Normalization attention
          and Residual block（深度可分离卷积 + 归一化注意力 + 残差块）

论文图 2 的两条支路（本项目 1:1 对应）：

    输入特征 x
      ├── 主支路（特征提取支路）
      │     深度卷积 → 归一化(BN) → [NAM门: NAM / 特征直通] → 逐点卷积(1×1) → 归一化(BN)
      │
      └── 残差支路（自适应残差支路）
            预判断：若输入输出通道数相同且步幅=1 → 特征直通(Identity)
                    否则 → 1×1 卷积调整通道数和尺寸（式(7)）

    融合（对应式(6)）：x_out = x_conv + x_identity

公式对照
--------
式(1)(2)   BatchNorm 归一化            → 在下面 ConvBNAct 内完成
式(3)(4)(5) NAM 通道注意力             → 见同目录 nam.py
式(6)      残差相加                    → forward 最后一行 y + self.shortcut(x)
式(7)      1×1 卷积调整 identity 分支  → self.shortcut = ConvBNAct(c1, c2, 1, s)

设计动机（为什么它能"轻量不掉点"）
----------------------------------
1. 深度可分离卷积把标准卷积拆成 "逐通道空间卷积 + 1×1 通道混合" 两步，
   参数量约为标准卷积的 1/8~1/9（docs/02 有手算例子）。
2. NAM 用 BN 的 γ 做通道注意力，零新增参数，专治小目标通道特征弱的问题。
3. 残差连接保证梯度顺畅流动、信息不丢失，让"轻量化"不以精度为代价。
4. "双支路选择策略"：NAM 门是可学习标量（sigmoid 门控），训练中自动学出
   NAM 该贡献多少；残差支路的"预判断"在构建期决定走直通还是 1×1 卷积。

说明：论文图 2 只画了 卷积-归一化，未显式画激活函数；本复现按深度网络常规
在卷积后加入 SiLU 激活（可用 use_act=False 关闭），这也是 YOLO 系列的默认做法。
"""

import torch
import torch.nn as nn

from .nam import NAMChannelAttention


class ConvBNAct(nn.Module):
    """最基础的三件套：Conv2d(bias=False) → BatchNorm2d → SiLU。

    等价于 ultralytics 里的 Conv 模块；这里自己实现是为了让本文件
    只依赖 PyTorch，便于零基础阅读。
    注意：卷积后接 BN 时卷积不需要 bias（BN 的 β 会把它抵消掉）——省参数。
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        # padding = k//2 保证 stride=1 时输出尺寸不变（same padding）
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)                     # 式(1)(2)
        self.act = nn.SiLU() if act else nn.Identity()   # SiLU(x) = x·sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DSNRblock(nn.Module):
    """DSNR-YOLO 的特征提取单元，用于替换 YOLOv11 主干中的 C3k2。

    Parameters
    ----------
    c1 : int  输入通道数
    c2 : int  输出通道数
    s  : int  步幅（论文主干中 DSNRblock 不负责下采样，下采样交给 GhostConv，默认 1）
    use_nam : bool  是否启用 NAM 门控双支路（False 时退化为纯深度可分离卷积 + 残差）
    use_act : bool  是否使用 SiLU 激活
    """

    def __init__(self, c1: int, c2: int, s: int = 1, use_nam: bool = True, use_act: bool = True):
        super().__init__()
        # ---------- 主支路：特征提取支路 ----------
        # ① 深度卷积 + 归一化（groups=c1 即"每个通道单独卷"，是轻量的关键）
        self.dw = ConvBNAct(c1, c1, k=3, s=s, g=c1, act=use_act)

        # ② NAM 门：门控值 g=sigmoid(θ) 可学习，θ 初始为 0 → 初始 50%/50% 混合，
        #    训练中自动学出"NAM 应该贡献多少"（双支路选择策略）
        self.nam = None
        self.nam_gate = None
        if use_nam:
            self.nam = NAMChannelAttention(self.dw.bn)   # 复用深度卷积那层 BN 的 γ
            self.nam_gate = nn.Parameter(torch.zeros(1))

        # ③ 逐点卷积（1×1）+ 归一化：负责跨通道信息融合与通道数变换
        self.pw = ConvBNAct(c1, c2, k=1, s=1, act=use_act)

        # ---------- 残差支路：预判断 ----------
        if c1 != c2 or s != 1:
            # 式(7)：通道数不同或步幅≠1 时，用 1×1 卷积自动调整（act=False 即只有卷积+BN）
            self.shortcut = ConvBNAct(c1, c2, k=1, s=s, act=False)
        else:
            # 特征直通：形状完全一致，直接原样传过去
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 主支路 ---
        y = self.dw(x)                                   # 深度卷积-归一化
        if self.nam is not None:                         # NAM 门关（选择策略）
            g = torch.sigmoid(self.nam_gate)             # 门控系数 ∈ (0,1)
            y = (1.0 - g) * y + g * self.nam(y)          # 特征直通 ⊕ NAM 加权混合
        y = self.pw(y)                                   # 逐点卷积-归一化
        # --- 融合：式(6) 残差相加 ---
        return y + self.shortcut(x)
