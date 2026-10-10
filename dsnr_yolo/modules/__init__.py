"""DSNR-YOLO 自定义模块集合。

- NAMChannelAttention : 论文式(3)-(5)，基于 BN γ 的通道注意力
- DSNRblock           : 论文图2，深度可分离卷积 + NAM门 + 残差 的双支路模块
- BiFPNFusion         : 论文图4，BiFPN 快速归一化加权融合节点
"""

from .nam import NAMChannelAttention
from .dsnr_block import DSNRblock, ConvBNAct
from .bifpn import BiFPNFusion

__all__ = ["NAMChannelAttention", "DSNRblock", "ConvBNAct", "BiFPNFusion"]
