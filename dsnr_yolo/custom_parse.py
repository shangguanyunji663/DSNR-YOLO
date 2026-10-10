"""DSNR-YOLO 专用的 parse_model（网络解析器）

背景知识（详见 docs/08）
------------------------
ultralytics 不用 Python 代码堆网络，而是读 yaml 逐层"搭积木"：
每行 [from, repeats, module, args]，由 parse_model 负责把 module 名字变成类、
并根据 width/depth 系数算好通道数后实例化。原版 parse_model 内部硬编码了
"它认识的模块该怎么算通道"，不认识 DSNRblock / BiFPNFusion，所以本文件提供一个
同构的解析器：规则与原版一致，另外支持我们的自定义模块。

替换方式见 register.py：把 ultralytics 命名空间里的 parse_model 换成本文件
的 dsnr_parse_model，之后 DetectionModel 构建、YOLO 训练全部自动生效。
"""

import ast
import contextlib

import torch
from torch import nn

# ultralytics 的官方积木（直接复用，保证行为一致）
from ultralytics.nn.modules import Conv, GhostConv, SPPF, C2PSA, C3k2, Concat, Detect
from ultralytics.utils.ops import make_divisible

from .modules import DSNRblock, BiFPNFusion

# yaml 模块名 → 类 的注册表。想扩展自己的模块，往这里加一行即可。
MODULE_REGISTRY = {
    "Conv": Conv,
    "GhostConv": GhostConv,
    "SPPF": SPPF,
    "C2PSA": C2PSA,
    "C3k2": C3k2,
    "Concat": Concat,
    "Detect": Detect,
    "DSNRblock": DSNRblock,
    "BiFPNFusion": BiFPNFusion,
}


def resolve_module(name: str):
    """按名字找模块类：先查注册表，再试 torch.nn（如 nn.Upsample）。"""
    if name in MODULE_REGISTRY:
        return MODULE_REGISTRY[name]
    if name.startswith("nn."):
        return getattr(torch.nn, name[3:])
    raise KeyError(
        f"yaml 中出现未注册的模块名 '{name}'。"
        f"请在 dsnr_yolo/custom_parse.py 的 MODULE_REGISTRY 中注册它。"
    )


def dsnr_parse_model(d: dict, ch, verbose: bool = True):
    """解析 DSNR-YOLO 的 yaml，返回 (nn.Sequential 模型, 需要保存中间输出的层号列表)。

    d  : yaml 读进来的字典（含 nc / depth_multiple / backbone / head ...）
    ch : 输入通道数（通常为 3，即 RGB）
    """
    # ---- 1. 读取缩放系数（与 ultralytics 规则一致）----
    scales = d.get("scales")
    scale = d.get("scale")
    if scales and scale in scales:
        depth, width, max_ch = scales[scale]
    else:  # 老式写法：直接写 depth_multiple / width_multiple（本项目 yaml 用这种）
        depth = d.get("depth_multiple", 1.0)
        width = d.get("width_multiple", 1.0)
        max_ch = d.get("max_channels", 1024.0)
    nc = d.get("nc", 80)
    # 8.4 版 Detect 头需要这两个参数（与官方 parse_model 一致）
    reg_max = d.get("reg_max", 16)        # DFL 分布焦点损失的离散化格点数
    end2end = d.get("end2end", False)     # 是否 NMS-free 端到端模式
    legacy = True                         # v8/v9 旧式检测头；出现 C3k2 时按官方规则置 False

    ch = [ch] if isinstance(ch, int) else list(ch)   # ch 列表：记录每一层的输出通道数
    layers, save, c2 = [], [], ch[-1]

    if verbose:
        print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<22}{'arguments':<30}")

    # ---- 2. 逐层构建（backbone 与 head 拼在一起顺序处理）----
    for i, (f, n, m_name, args) in enumerate(d["backbone"] + d["head"]):
        m = resolve_module(m_name)

        # yaml 里的字符串参数需要转换：'nc' → 类别数；'None'/'True'/数字字符串 → Python 值
        #（yaml 中不带引号的 None 其实是字符串 "None"，官方解析器同样用 ast 转回来）
        for j, a in enumerate(args):
            if isinstance(a, str):
                if a == "nc":
                    args[j] = nc
                else:
                    with contextlib.suppress(ValueError):
                        args[j] = ast.literal_eval(a)

        # 重复次数按深度系数缩放（s 版 ×0.5，如 2 → 1）
        n = n_ = max(round(n * depth), 1) if n > 1 else n

        if m in (Conv, GhostConv, DSNRblock):
            # 规则：args[0] 是"名义输出通道"，乘宽度系数、夹到 max_ch、8 对齐
            c1 = ch[f]
            c2 = args[0]
            c2 = make_divisible(min(c2, max_ch) * width, 8)
            args = [c1, c2, *args[1:]]

        elif m in (SPPF, C3k2, C2PSA):
            c1 = ch[f]
            c2 = make_divisible(min(args[0], max_ch) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in (C3k2, C2PSA):        # 这两类模块的 args[2] 是重复次数 n
                args.insert(2, n_)
                n = 1                     # n 已塞进模块参数，外层不再 Sequential 重复
            if m is C3k2:                 # 官方规则：出现 C3k2 即 v11 式新检测头
                legacy = False

        elif m is nn.Upsample:
            c2 = ch[f]                    # 上采样不改变通道数

        elif m is BiFPNFusion:
            # 多输入节点：所有输入通道必须一致；把 (通道数, 路数) 传给模块
            f_list = f if isinstance(f, list) else [f]
            c2 = ch[f_list[0]]
            args = [c2, len(f_list)]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)    # 拼接后通道数 = 各路之和

        elif m is Detect:
            # 8.4 版签名：Detect(nc, reg_max, end2end, ch=各输入层通道列表)
            args = [nc, reg_max, end2end, [ch[x] for x in f]]
            m.legacy = legacy             # 与官方 parse_model 一致
            c2 = nc

        else:
            c2 = ch[f]                    # 兜底：不改变通道数

        # ---- 实例化（n>1 时包成 Sequential）----
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)

        # ---- 记录元信息（ultralytics 的 model.info / 前向保存机制依赖这些属性）----
        t = str(m)[8:-2].replace("__main__.", "")          # 类的完整路径名
        m_.np = sum(x.numel() for x in m_.parameters())    # 该层参数量
        m_.i, m_.f, m_.type = i, f, t                      # 层号 / 输入来源 / 类型名
        # 需要缓存的中间层：非 -1 的输入来源（多输入时逐个登记）
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)

        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

        if verbose:
            print(f"{i:>3}{str(f):>18}{n_:>3}{m_.np:>10}  {t:<22}{str(args):<30}")

    model = nn.Sequential(*layers)
    save = sorted(list({x for x in save}))   # 去重排序
    if verbose:
        print(f"\n模型层数={len(layers)}, 输出通道={ch}, 需缓存层={save}")
    return model, save
