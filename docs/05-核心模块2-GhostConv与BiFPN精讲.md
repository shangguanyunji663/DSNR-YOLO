# 05 · 核心模块 2 —— GhostConv 与 BiFPN 精讲（论文 2.3/2.4 节）

> 这两个改进一个管"省"，一个管"补"：GhostConv 继续压参数，
> BiFPN 把轻量化丢掉的多尺度融合质量补回来。
> 配套动手：`tests/test_modules.py::test_ghost_conv / test_bifpn_fusion`。

---

## 一、GhostConv（2.3 节 + 图 3）

### 1. 原理回顾（docs/02 已推导，此处讲工程用法）

```
输入 ──[正常卷积 → C_out/2]──┬── 本征特征
                             └──[深度可分离 5×5 → C_out/2]── 幽灵特征
输出 = concat(本征, 幽灵)    # 通道数不变，参数≈2 倍压缩
```

论文图3 的示意即此：Conv 主路 + Identity 廉价支路，再拼接。

### 2. 在论文里的两个部署位置（对照图1）

| 位置 | 图1 层号 | 替换了什么 | 为什么 |
|---|---|---|---|
| 主干下采样 | 3 / 5 / 7 | `Conv[c,3,2]` | 下采样卷积通道宽（128/256/512），参数密集 |
| 颈部下采样 | 20 / 23 / 26 | PAN 的下采样 Conv | 同上，且颈部本就该轻 |

本项目 yaml 中对应行（`dsnr_yolo/models/dsnr_yolo.yaml`）：

```yaml
- [-1, 1, GhostConv, [256, 3, 2]]   # 3  P3/8 下采样
- [-1, 1, GhostConv, [512, 3, 2]]   # 5  P4/16 下采样
- [-1, 1, GhostConv, [1024, 3, 2]]  # 7  P5/32 下采样
- [-1, 1, GhostConv, [512, 3, 2]]   # 20/23 颈部下采样
```

> GhostConv 是 ultralytics **内置模块**（`ultralytics.nn.modules.GhostConv`），
> yaml 里直接写名字即可 —— 不需要自己实现，也不需要注册。参数实测对比见
> tests 的 `test_ghost_conv`（≈1.9 倍压缩）。

### 3. 表2 中它贡献了什么

模型1→模型2（加 GhostConv）：8.00M→6.69M（−1.31M），18.9→16.1 GFLOPs（−2.8），
mAP50 88.4→87.8（−0.6）。**注意 mAP 是降的** —— 轻量化的代价由下一个改进（BiFPN）
来补偿，这正是论文"三位一体"设计的原因。

---

## 二、BiFPN（2.4 节 + 图 4）

### 1. 为什么等权融合是缺陷

FPN/PAN 的融合是**无参数的加法/拼接**：三路特征"一视同仁"。
但遥感图像里：
- 小目标的特征在**浅层高分辨率**特征图（P3）上，通道占比极低；
- 深层 P5 语义强但小目标早已被压缩稀释，甚至充满背景噪声；
- 等权相加 = 让噪声稀释信号。

BiFPN 的解法：**每个融合节点为每路输入学一个权重**：

```
w_i' = ReLU(w_i) / (Σ_j ReLU(w_j) + ε)      快速归一化（保证非负、和为1、无指数运算）
out  = SiLU( Σ_i w_i' × x_i )
```

对比 softmax 归一化：效果相近但省掉 exp 运算，训练更稳（EfficientDet 原文结论）。

### 2. 图4 解读与本项目拓扑

论文图4 是 4 尺度（P2~P5）BiFPN：横向箭头=同层融合，红色斜箭头=跨层加权连接。
本项目按**三尺度 P3~P5 的标准 EfficientDet 拓扑**复现（yaml head，11~26 层）：

```
侧向对齐:  10 ──1×1── 11 (P5,256ch)     6 ──1×1── 12 (P4,256ch)     4 ──1×1── 13 (P3,128ch)
                                    │
自顶向下:  11 ──Upsample── 14 ─┐    │
                        15 ⊕←─┼────┘   (P4 顶向下融合: 12 + 14)
                        16 C3k2→128ch
                        17 Upsample
                        18 ⊕←─ 13 + 17          (P3 融合 → P3 输出)
                        19 C3k2 → 128ch         = P3 输出特征
自底向上:  19 ──GhostConv↓2── 20
           21 ⊕←─ 12 + 15 + 20              (P4 输出节点: 三路加权!)
           22 C3k2 → 256ch                  = P4 输出特征
           22 ──GhostConv↓2── 23
           24 ⊕←─ 11 + 23                   (P5 输出节点)
           25 C3k2 → 512ch                  = P5 输出特征
检测头:    26 Detect([19, 22, 25])
```

> ⊕ = BiFPNFusion（加权融合）。21 号节点吃三路输入（侧向 + 顶向下中间结果 +
> 下采样的浅层输出）—— 这是 BiFPN 区别于 PAN 的标志，图4 中每个尺度都有这样
> 的"重复利用"节点。

### 3. 融合节点代码（`dsnr_yolo/modules/bifpn.py`）

```python
class BiFPNFusion(nn.Module):
    def __init__(self, c, n=2, eps=1e-4):
        self.w = nn.Parameter(torch.ones(n))   # 每路一个可学习权重

    def forward(self, x):                       # x: list[Tensor]，各路通道数必须相同
        w = F.relu(self.w)                      # ReLU 保证非负（负权重无意义）
        w = w / (w.sum() + self.eps)            # 快速归一化
        out = Σ w_i * x_i
        return self.act(out)                    # SiLU
```

要点：
- **通道对齐**在融合前的 1×1 Conv（yaml 11/12/13）完成，融合节点只做加权；
- **输入路数 n** 由 `custom_parse.py` 从 yaml 的 from 列表自动推断传入；
- 4 个融合节点、每个只有 2~3 个权重参数 —— BiFPN 的参数开销几乎可以忽略，
  它买的是"融合质量"。

### 4. 复现说明（诚实声明）

论文图1 颈部共 18 层（11~28，5 个 BiFPN 节点 + 5 个 C3k2），但**未公开完整连线**
与各层通道配置。本复现按 EfficientDet 规范实现为 16 层（4 个融合节点，其中
P4 输出为三输入节点），通道阶梯与 v11 一致（P3=128/P4=256/P5=512）。
构建后实测 **6.01M 参数**（论文 7.15M，差异主要来自论文未公开的颈部细节）。

想自行逼近 7.15M？把 yaml head 里各 C3k2 通道参数加大即可（例如 P4 级
`[512,...]→[1024,...]`），重跑 `scripts/verify_model.py` 观察 —— 这本身就是
很好的架构实验练习。

### 5. 训练后看一眼融合权重（可解释性）

```python
import torch
ckpt = torch.load('runs/dsnr_yolo/weights/best.pt', map_location='cpu', weights_only=False)
for i, layer in enumerate(ckpt['model'].model):
    if type(layer).__name__ == 'BiFPNFusion':
        w = torch.relu(layer.w); w = w / (w.sum() + layer.eps)
        print(f'层{i} 融合权重: {[round(x,3) for x in w.tolist()]}')
```

如果 P3 相关节点的浅层权重明显更大，就验证了论文"主动聚焦小目标特征"的说法。

---

## 本章自测

1. GhostConv 的"幽灵特征"为什么可以用廉价操作生成？
2. BiFPN 融合公式里 ReLU 和 ε 各防止什么？
3. 为什么 P4 输出节点能吃三路输入而 P3 只有两路？（提示：P3 是金字塔最底层，
   没有更浅的层可再融合）
4. 本复现的 BiFPN 与论文图4 的 P2~P5 四尺度版本差在哪？什么时候需要 P2？

---

*下一篇 → [06-核心模块3：自适应混合损失函数精讲](06-核心模块3-自适应混合损失函数精讲.md)*
