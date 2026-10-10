"""DSNR-YOLO 的 DetectionModel 子类

为什么要子类化 ultralytics 的 DetectionModel？
    默认的 DetectionModel.init_criterion() 会返回原版 v8DetectionLoss
    （等权 BCE 分类损失）。论文 2.5 节要求换成"ATFL+SLIDE 可学习加权"的
    混合损失，所以在这里重写 init_criterion：
      1. 返回我们的 MixedV8DetectionLoss（见 losses/mixed_loss.py）；
      2. 用 add_module 把损失挂成模型的子模块 —— 这样损失里的可学习权重
         ω1、ω2 会自动进入 model.parameters()，从而被优化器更新。
"""

from ultralytics.nn.tasks import DetectionModel


class DSNRDetectionModel(DetectionModel):
    """携带混合损失的 YOLO 检测模型（结构、前向、保存与原版完全一致）。"""

    def init_criterion(self):
        """训练器首次调用 model.loss() 时会被触发一次。"""
        from .losses import MixedV8DetectionLoss

        criterion = MixedV8DetectionLoss(self)
        # 关键一步：注册为子模块 → ω1/ω2 出现在 model.parameters() 里 → 会被优化器更新
        self.add_module("_dsnr_criterion", criterion)
        return criterion
