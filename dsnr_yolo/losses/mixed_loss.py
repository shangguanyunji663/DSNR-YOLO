"""自适应混合损失函数 MixedV8DetectionLoss（论文 2.5 节 式(8)–(11) 的实现）

设计总览（先读 docs/06 再看代码会更顺）
================================================================
YOLOv11 原版分类损失对所有样本"等权惩罚"（式(8) 的 BCE）。
论文将其替换为两路"各管一件事"的调制损失再加权求和：

  ① ATFL（式10）：难度自适应 —— 分类越难的样本权重越大（focal 思想）
       ATFL_loss = loss_bce × modulating_factor,   modulating_factor = (1-pt)^γ
  ② SLIDE（式9）：质量分层 —— IoU 置信度越低（定位越差）的正样本权重越大
       SLIDE_loss = loss_bce × modulating_weight,  w = exp(-μ(pt-θ))，θ=batch 正样本平均IoU
  ③ 可学习加权混合（式11）：
       total_cls = ω1·ATFL + ω2·SLIDE + α·ω1² + β·ω2²
       其中 ω1、ω2 是【可学习参数】（进入优化器自动更新）；
       α、β 是正则化系数，防止 ω 无限增大（论文原文："防止损失的权重无限增大"）。

说明：论文正文只给出了抽象形式（式9-11），未公开具体系数；本复现按
ATFL（EFLNet, TGRS 2024）与 SLIDE（YOLO-FaceV2, PR 2024）原始论文的思想实现，
系数取 γ=1.5、μ=0.5、α=β=0.01，均可在 __init__ 里调整。

实现方式：继承本机安装的 ultralytics v8DetectionLoss（8.4.164），只重写
get_assigned_targets_and_loss 中的【分类损失】一段；框回归(CIoU)与分布焦点
损失(DFL) 保持原版不动 —— 论文只改了"等权惩罚"的分类部分。
"""

from __future__ import annotations

import torch
from torch import nn

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class MixedV8DetectionLoss(nn.Module, v8DetectionLoss):
    """YOLOv8/v11 检测损失 + 论文的 ATFL/SLIDE 可学习加权分类损失。

    同时继承 nn.Module 的原因：
      1. 让 ω1/ω2 成为真正的 nn.Parameter，能被 add_module 挂进模型、
         从而进入优化器（原版 v8DetectionLoss 是普通类，做不到）；
      2. nn.Module 的 __call__ 会调用 forward，所以推理入口写在 forward 里。
    """

    def __init__(self, model, tal_topk: int = 10):
        nn.Module.__init__(self)                                   # 先初始化 Module 结构
        v8DetectionLoss.__init__(self, model, tal_topk=tal_topk)   # 再初始化原版（bce/assigner/bbox_loss）
        # ---- 式(11) 的两个可学习精度权重（会注册进优化器）----
        self.w_atfl = nn.Parameter(torch.tensor(1.0))    # ω1：ATFL 的权重
        self.w_slide = nn.Parameter(torch.tensor(1.0))   # ω2：SLIDE 的权重
        # ---- 超参数（论文未公开，取常用值，可调整）----
        self.gamma = 1.5        # ATFL 聚焦系数 γ：越大越"偏心"难样本
        self.mu = 0.5           # SLIDE 形状参数 μ：越大质量分层越陡峭
        self.reg_alpha = 0.01   # 正则化强度 α（式11 中 α=β，共用一个系数）
        # 损失名加注，训练日志里能看到
        self.loss_names = ("box_loss", "cls_mixed_loss", self.loss_names[-1])

    def forward(self, preds, batch):
        """与原版 v8DetectionLoss.__call__ 相同的入口（nn.Module 的调用约定走 forward）。"""
        return self.loss(self.parse_output(preds), batch)

    def get_assigned_targets_and_loss(self, preds: dict, batch: dict) -> tuple:
        """复制自 ultralytics 8.4.164 的 v8DetectionLoss，仅分类损失一段被替换。

        标注 [原版] 的行与官方源码一致；标注 [论文] 的行是本次修改。
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)  # [原版]

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # ---- 把 batch 里的真实框整理成 (B, 最大目标数, 5) 的张量 ----[原版]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # ---- Task-Aligned 动态标签分配：给每个 anchor 点分配合适的 GT ----[原版]
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp_(min=1)

        # ==================== 分类损失：论文式(8)→(11) ====================
        # 式(8)：基础二元交叉熵（逐元素），target_scores 是 0~1 的软标签
        #（v8 系列把 IoU 质量编码进分类标签，所以它同时是"IoU 置信度"）
        bce = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce = bce * self.class_weights

        t = target_scores.to(dtype)
        # pt：模型对"正确答案"的置信度（分类越准 pt 越接近 1）
        p = pred_scores.sigmoid()
        pt = p * t + (1.0 - p) * (1.0 - t)

        # ---- 式(10) ATFL：难度自适应调制因子 (1-pt)^γ ----
        # pt 小 → 难样本 → (1-pt)^γ 接近 1，权重保留；pt 大 → 简单样本 → 权重趋近 0
        atfl = bce * (1.0 - pt).pow(self.gamma)

        # ---- 式(9) SLIDE：IoU 感知调制权重（质量分层）----
        # θ：本 batch 正样本的平均 IoU 置信度（质量阈值），detach 不参与梯度
        pos = target_scores > 0
        if pos.any():
            theta = target_scores[pos].mean().detach()
        else:
            theta = target_scores.new_tensor(0.25)
        # w = exp(-μ(pt-θ))：质量低于阈值(pt<θ) → w>1 放大；高于阈值 → w<1 缩小
        slide_w = torch.exp(-self.mu * (pt - theta))
        slide = bce * slide_w

        # ---- 式(11)：可学习加权混合 + 正则化约束（α·ω1² + β·ω2²）----
        cls_mixed = (self.w_atfl * atfl + self.w_slide * slide).sum() / target_scores_sum
        reg = self.reg_alpha * (self.w_atfl ** 2 + self.w_slide ** 2)
        loss[1] = cls_mixed + reg
        # ==================================================================

        # ---- 框回归 CIoU + 分布焦点 DFL：与原版完全一致 ----[原版]
        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )
