"""把 DSNR-YOLO 的自定义内容"注入"ultralytics（本项目最核心的胶水代码）

只需在训练/推理脚本开头调用一次 apply()，之后：
    from ultralytics import YOLO
    model = YOLO("dsnr_yolo/models/dsnr_yolo.yaml")   # ← 就能认识 DSNRblock/BiFPNFusion

打补丁的三个点（原理详见 docs/08）：
  ① parse_model：原版解析器不认识我们的自定义模块，换成 custom_parse.py 的版本。
  ② DetectionTrainer.get_model：训练器在正式训练前会"按 yaml 重新建一个模型"，
     在这里把普通 DetectionModel 换成带混合损失的 DSNRDetectionModel（仅当
     yaml 里有 dsnr: true 标记，保证不影响官方 baseline 模型）。
  ③ make_toy 之类的脚本无需补丁；预测/验证直接复用 ①。

所有补丁只在当前 Python 进程内生效，不修改 ultralytics 安装文件本身。
"""

import inspect

_APPLIED = False


def apply(verbose: bool = True):
    """应用全部注册补丁（幂等：重复调用只生效一次）。"""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    # ---- 补丁①：替换 parse_model ----
    import ultralytics.nn.tasks as tasks
    from .custom_parse import dsnr_parse_model
    from .modules import DSNRblock, NAMChannelAttention, BiFPNFusion

    tasks_mod = inspect.getmodule(tasks.DetectionModel)   # DetectionModel 真正定义处
    tasks_mod.parse_model = dsnr_parse_model
    # 顺手把自定义类也注册进 tasks 命名空间（部分版本按名字反射时用得到）
    for cls in (DSNRblock, NAMChannelAttention, BiFPNFusion):
        setattr(tasks_mod, cls.__name__, cls)
    if verbose:
        print("[DSNR-YOLO] 补丁①：parse_model 已替换为 dsnr_parse_model")

    # ---- 补丁②：训练器建模型时，识别 dsnr 标记并换成自定义模型类 ----
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from .dsnr_model import DSNRDetectionModel

    _orig_get_model = DetectionTrainer.get_model

    def _dsnr_get_model(self, cfg=None, weights=None, verbose=True):
        model = _orig_get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        # 构建出的模型带有 .yaml 属性；带 dsnr: true 标记的才换成我们的类
        # （cfg 可能是路径字符串也可能是字典，看模型自己的 yaml 最稳妥）
        yaml_dict = getattr(model, "yaml", None)
        if isinstance(yaml_dict, dict) and yaml_dict.get("dsnr", False):
            model.__class__ = DSNRDetectionModel
        return model

    DetectionTrainer.get_model = _dsnr_get_model
    if verbose:
        print("[DSNR-YOLO] 补丁②：DetectionTrainer.get_model 已挂钩（识别 dsnr 标记）")

    # ---- 补丁③：构建优化器前，先创建混合损失 ----
    # 原版 ultralytics 在第一个 batch 才懒创建损失；那时优化器已经建好，
    # 混合损失里的可学习权重 ω1/ω2 会漏在优化器之外。这里在 build_optimizer
    # 之前提前创建损失（其参数已通过 add_module 注册进模型），确保 ω1/ω2 被优化。
    from ultralytics.engine.trainer import BaseTrainer
    from .dsnr_model import DSNRDetectionModel

    _orig_build_optimizer = BaseTrainer.build_optimizer

    def _dsnr_build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        m = model.module if hasattr(model, "module") else model  # 兼容 DDP 包装
        if isinstance(m, DSNRDetectionModel) and getattr(m, "criterion", None) is None:
            m.criterion = m.init_criterion()
        return _orig_build_optimizer(
            self, model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations
        )

    BaseTrainer.build_optimizer = _dsnr_build_optimizer
    if verbose:
        print("[DSNR-YOLO] 补丁③：build_optimizer 已挂钩（混合损失权重将参与训练）")


def swap_class(model):
    """把已构建好的 YOLO 实例内部的模型换成 DSNRDetectionModel。

    用于 YOLO("xxx.yaml") 直接构建的场景（不经过 trainer）。
    """
    from .dsnr_model import DSNRDetectionModel

    model.model.__class__ = DSNRDetectionModel
    return model


def is_dsnr(model) -> bool:
    """检查一个 YOLO/DetectionModel 实例是否已是 DSNR 模型。"""
    return type(model.model if hasattr(model, "model") else model).__name__ == "DSNRDetectionModel"
