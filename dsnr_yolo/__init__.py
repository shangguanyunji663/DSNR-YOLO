"""DSNR-YOLO：轻量化注意力残差增强 YOLO（论文复现包）

用法（在任何脚本入口）：
    from dsnr_yolo import register
    register.apply()          # 打补丁：注册自定义模块 + 混合损失
    from ultralytics import YOLO
    model = YOLO("dsnr_yolo/models/dsnr_yolo.yaml")
    register.swap_class(model)   # 换成带混合损失的模型类
"""

from .register import apply, swap_class, is_dsnr

__all__ = ["apply", "swap_class", "is_dsnr"]
