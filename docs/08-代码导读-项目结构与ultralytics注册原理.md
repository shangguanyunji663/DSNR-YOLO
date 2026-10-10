# 08 · 代码导读 —— 项目结构与 ultralytics 注册原理

> 本篇回答两个问题：① 每个文件是干嘛的、张量怎么流过网络；
> ② 为什么"魔改 YOLO"需要打补丁、补丁打在哪。
> 这是全套文档里"工程含金量"最高的一篇，读前建议先跑通
> `python scripts/verify_model.py` 并对着输出看。

---

## 1. 一次训练的完整调用链

```
你运行:  python scripts/train.py --data data/dior.yaml --epochs 200
   │
   ├─① import dsnr_yolo; register.apply()
   │      打三个补丁（见第 3 节）—— 必须在 import ultralytics 的模型构建之前
   │
   ├─② YOLO("dsnr_yolo/models/dsnr_yolo.yaml")
   │      ultralytics 读 yaml → 调用【我们的】parse_model 逐层建网
   │      （DSNRblock/BiFPNFusion 从注册表按名字找到类）
   │
   ├─③ register.swap_class(model)
   │      把模型对象换成 DSNRDetectionModel 类（携带混合损失的 init_criterion）
   │
   ├─④ model.train(...)
   │      ultralytics 训练器接管：数据增强/优化器/日志/验证
   │      首次 loss 时 → init_criterion() → MixedV8DetectionLoss
   │      ω₁/ω₂ 已被补丁③提前创建 → 进入优化器参数组
   │
   └─⑤ 每轮结束验证 + 保存 best.pt / last.pt
```

## 2. 张量形状追踪（640×640 输入，v11s 宽度系数 0.5）

| 层 | 模块 | 输出形状 (B,C,H,W) | 备注 |
|---|---|---|---|
| 0 | Conv 3,2 | 32×320×320 | P1/2 |
| 1 | Conv 3,2 | 64×160×160 | P2/4 |
| 2 | DSNRblock | 128×160×160 | 通道 64→128，残差走 1×1 |
| 3 | GhostConv 3,2 | 128×80×80 | P3/8 |
| 4 | DSNRblock | 256×80×80 | P3 侧向特征 |
| 5 | GhostConv 3,2 | 256×40×40 | P4/16 |
| 6 | DSNRblock | 256×40×40 | P4 侧向特征 |
| 7 | GhostConv 3,2 | 512×20×20 | P5/32 |
| 8 | DSNRblock | 512×20×20 | 残差走 Identity（通道不变） |
| 9~10 | SPPF, C2PSA | 512×20×20 | v11 原样 |
| 11~13 | Conv 1×1 | 256/256/128 | 三路侧向对齐 |
| 14,17 | Upsample | 尺寸×2 | 最近邻插值 |
| 15,18,21,24 | BiFPNFusion | 与输入同 | 可学习加权求和 |
| 16,19,22,25 | C3k2 | … | 融合后特征提炼 |
| 20,23 | GhostConv 3,2 | 尺寸÷2 | 自底向上路径 |
| 26 | Detect | 3 尺度预测 | 训练返回 dict{boxes,scores,feats} |

（`scripts/verify_model.py` 的输出就是这个表的机器版。）

## 3. 三大补丁的原理（register.py）

ultralytics 是"yaml 驱动"的框架：不认识你的自定义模块，就得告诉它。
本项目**不修改** ultralytics 安装文件，全部用进程内 monkey-patch：

### 补丁①：替换 parse_model

`ultralytics.nn.tasks` 里的 `parse_model` 负责把 yaml 每行
`[from, repeats, module, args]` 变成实例，内部硬编码了每种模块的通道规则。
我们用 `custom_parse.py::dsnr_parse_model` 替换它：

- 与原版同样的缩放规则（width/depth/max_channels/8 对齐）；
- 增加了两条自定义规则：
  - `DSNRblock` 同 Conv/GhostConv：`args[0]` = 名义输出通道；
  - `BiFPNFusion`：从 from 列表推断输入路数，检查各路通道一致；
- 模块名解析：先查 `MODULE_REGISTRY`，`nn.` 开头再去 `torch.nn` 找。

```python
tasks_mod = inspect.getmodule(tasks.DetectionModel)
tasks_mod.parse_model = dsnr_parse_model   # DetectionModel 构建时按名查找 → 生效
```

### 补丁②：训练器建模后换类

训练器 `DetectionTrainer.get_model` 按 cfg 重新建模型。我们包一层：
建完后看 `model.yaml` 里有没有 `dsnr: true` 标记，有就把
`model.__class__ = DSNRDetectionModel`（Python 支持给实例换类！）。
好处：官方 yolo11s 等基线模型完全不受影响，同一进程可做对比实验。

### 补丁③：优化器前先建损失

原版 ultralytics 在**第一个 batch** 才懒创建 criterion，那时优化器已构建完毕，
损失里的 ω₁/ω₂ 会漏在优化器外（=永远不更新）。补丁③在
`BaseTrainer.build_optimizer` 前先 `m.criterion = m.init_criterion()`。
ω 参数名落在 "weight(with decay)" 参数组（名字不含 bias、模块不是 BN）。

> 验证三个补丁都生效：任何训练日志开头会打印
> `[DSNR-YOLO] 补丁①/②/③ ...`；训练后
> `model.model._dsnr_criterion.w_atfl != 1.0` 说明 ω 确实在被优化。

## 4. 混合损失的调用链（ loss 的一次完整流动）

```
trainer 循环
  → preds = model(img)                    # 训练模式 Detect 返回 dict
  → loss, items = model.loss(batch, preds)
      → criterion.forward = MixedV8DetectionLoss.forward
          → parse_output: 兼容 tuple/dict 两种预测格式
          → v8DetectionLoss.loss（原版）
              → get_assigned_targets_and_loss  ←【我们重写的方法】
                  ① make_anchors: 生成 8400 个 anchor 中心点
                  ② preprocess: 真框从 batch 整理成 (B, N_max, 5)
                  ③ assigner(TaskAligned): 给每个 anchor 配标签
                  ④ 式(8) BCE → 式(10) ATFL → 式(9) SLIDE → 式(11) 混合
                  ⑤ bbox_loss: CIoU + DFL（原版未动）
  → loss.sum() / batch_size 反传 → optimizer_step（梯度裁剪+EMA）
```

## 5. 常见魔改手法对照（你以后自己改时的套路）

| 想做的事 | 改哪里 |
|---|---|
| 加一个新模块（如新的注意力） | `modules/` 新建类 → `custom_parse.py` 注册表加一行 → yaml 里用 |
| 改网络深度/宽度 | yaml 头部 `depth_multiple / width_multiple` |
| 换/调损失系数 | `losses/mixed_loss.py` 的 `__init__` |
| 通道数实验 | yaml 各层 args（第一个数字） |
| 做消融（去掉某改进） | yaml 里把对应模块名换回原版（DSNRblock→C3k2 等） |
| 基线对比 | `YOLO('yolo11s.yaml')` 直接训练，补丁不影响它 |

## 6. 设计原则复盘（为什么这样组织代码）

1. **模块代码零依赖 ultralytics**（`modules/` 只有 torch）——便于单元测试、
   便于移植到其它框架；
2. **框架胶水集中在一个文件**（register.py）——升级 ultralytics 时只查这一处；
3. **所有"论文未公开"的决策都写成可调参数+注释**（门控形式、损失系数、
   颈部宽度）——复现的可辩驳性比"看起来一样"重要；
4. **yaml 即文档**——每行带论文章节号注释，结构改动一目了然。

---

*下一篇 → [09-复现指南：环境、数据、训练与调参](09-复现指南-环境数据训练与调参.md)*
