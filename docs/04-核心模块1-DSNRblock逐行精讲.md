# 04 · 核心模块 1 —— DSNRblock 逐行精讲（论文 2.2 节 + 图 2）

> 论文最核心的创新。本文档把图2 的每个框、公式(1)-(7) 的每个符号
> 和 `dsnr_yolo/modules/dsnr_block.py` 的每行代码三者一一对应。
> 配套动手：`python tests/test_modules.py`（前 6 个测试）。

---

## 1. 模块要解决什么问题

主干特征提取模块（v11 里是 C3k2）参数多、算力贵。直接换成深度可分离卷积
（DepSep）虽然轻，但特征提取能力下降（论文表3：DepSep 版 mAP 从 66.2 掉到 65.3）。

论文的答案 = DepSep + NAM + 残差，用"双支路选择策略"组织起来，即 **DSNRblock**：

```
Depthwise separable convolution   深度可分离卷积      → 省参数
+ Normalization-based attention   归一化注意力(NAM)   → 保精度
+ Residual block                  残差连接            → 防退化
```

## 2. 图 2 数据流（与代码逐行对照）

论文图2（重画）：

```
              输入特征 x
        ┌─────────┴──────────────┐
        │                        │
  【特征提取支路】            【自适应残差支路】
        │                        │
  ① 深度卷积-归一化          预判断◇：
        │                    (c1==c2 且 s==1)?
  ② NAM门关◇                    │           │
   是│      │否               是↓          ↓否
   特征直通   NAM              特征直通    维度调整
        │      │             (Identity)  (1×1 Conv, 式7)
        └──┬───┘                  │           │
  ③ 逐点卷积-归一化                │           │
        │                        │           │
        └──────────┬─────────────┘
                特征融合 ⊕   ← 式(6)：x_out = x_conv + x_identity
                   │
                 输出
```

代码（`dsnr_yolo/modules/dsnr_block.py`，删去注释后的骨架）：

```python
class DSNRblock(nn.Module):
    def __init__(self, c1, c2, s=1, use_nam=True, use_act=True):
        # 【主支路①】深度卷积-归一化：groups=c1 → 每个通道独立 3×3 卷积
        self.dw = ConvBNAct(c1, c1, k=3, s=s, g=c1, act=use_act)

        # 【NAM门关】可学习标量门 θ，初始 0 → sigmoid(0)=0.5，NAM/直通各占一半；
        # 训练中自动学出 NAM 该贡献多少 —— 这就是论文的"双支路选择策略"
        self.nam = NAMChannelAttention(self.dw.bn)   # 复用①里 BN 的 γ
        self.nam_gate = nn.Parameter(torch.zeros(1))

        # 【主支路③】逐点卷积-归一化：1×1 卷积负责跨通道融合 + 通道数变换
        self.pw = ConvBNAct(c1, c2, k=1, s=1, act=use_act)

        # 【残差支路：预判断】通道数不同或步幅≠1 → 用 1×1 卷积调整（式7）
        self.shortcut = ConvBNAct(c1, c2, k=1, s=s, act=False) \
                        if (c1 != c2 or s != 1) else nn.Identity()

    def forward(self, x):
        y = self.dw(x)                     # ① 深度卷积-归一化
        g = torch.sigmoid(self.nam_gate)   # 门控系数 ∈ (0,1)
        y = (1-g)*y + g*self.nam(y)        # ② 直通 ⊕ NAM 的凸组合
        y = self.pw(y)                     # ③ 逐点卷积-归一化
        return y + self.shortcut(x)        # 式(6) 残差融合
```

## 3. 公式逐个落地

### 式(1)(2)：归一化 —— 在 ConvBNAct 里

```python
self.bn = nn.BatchNorm2d(c2)          # μ、σ、γ、β 都是它的内部状态/参数
self.act = nn.SiLU()
forward: act(bn(conv(x)))             # 先按式(1)标准化，再按式(2)缩放平移
```

每个通道维护 4 个量：μ、σ²（前向统计，无梯度）、γ、β（可学习，有梯度）。

### 式(3)(4)(5)：NAM —— `nam.py`

```python
w = bn.weight.abs()                   # |γ|，形状 (C,)
w = w / (w.sum() + eps)               # 式(3)：归一化成和为1的分布
x_att = w.view(1,-1,1,1) * x          # 式(4)：逐通道加权（广播）
out = sigmoid(x_att) * x              # 式(5)：门控
```

关键设计：**NAM 不新建任何 BN**，直接引用深度卷积那层 BN 的 γ。
所以整个 NAM 零新增参数（tests 里 `n_new_params == 0` 断言验证了这点）。

### 式(6)：残差相加

```python
return y + self.shortcut(x)
```

### 式(7)：1×1 卷积调整 identity

```python
self.shortcut = ConvBNAct(c1, c2, k=1, s=s, act=False)   # 仅当形状不匹配时
```

论文原话："如果输入和输出通道数不同或步幅不为 1，则会自动使用 1×1 卷积调整
输入特征图的通道数和尺寸"。对应主干里 DSNRblock 2（64→128 通道）与
DSNRblock 4（128→256 通道）走 1×1，DSNRblock 6/8（通道不变）走 Identity。

## 4. 参数量核算：省在哪、省多少

以主干第 8 层（512→512，v11s 配置）为例：

```
C3k2（v11 原版）：≈ 1.38 M
DSNRblock：
  深度卷积 512×3×3          = 4 608   (+BN 1024)
  NAM                       = 0
  逐点卷积 512×512×1×1      = 262 144 (+BN 1024)
  残差支路 Identity          = 0
  合计 ≈ 0.269 M                        → 单层省 5 倍
```

（实际四个 DSNRblock 总共 0.42M，被替换的四个 C3k2 总共 1.86M ——
对应论文表2 模型1：9.43M → 8.00M。）

> 注意残差支路的 1×1 卷积只在"通道变化"时出现：DSNRblock 2 (64→128) 8.4K、
> DSNRblock 4 (128→256) 33.6K；通道不变时是免费的 Identity。

## 5. 设计上值得咀嚼的三个细节

1. **门控为什么可学习？** 论文说"双支路选择策略"+"NAM门关"。
   静态开关（bool）无法适应不同层、不同数据对注意力的需求差异；
   可学习标量 g = sigmoid(θ) 让每个 block 自己决定"注意力掺多少"。
   代价：每 block 仅 +1 参数。（这也是复现中合理的自由度——论文未明确门的形式。）
2. **为什么 NAM 放在 DW 之后、PW 之前？** DW 之后特征已按通道独立提取，
   此时做通道加权正好"挑出好通道"，再由 PW 混合放大；若放在 PW 之后，
   混合后的通道权重语义更模糊。
3. **残差里为什么 act=False（只有 Conv+BN）？** 残差支路只负责"形状对齐"，
   引入非线性反而破坏 x_conv + x_identity 的线性可加性（标准 ResNet 同款处理）。

## 6. 复现与论文的两处自由度（透明声明）

| 事项 | 论文 | 本复现 | 理由 |
|---|---|---|---|
| NAM 门的形式 | 未明确 | 可学习标量 sigmoid 门 | "双支路选择"的最简可训练实现 |
| 激活函数 | 图2未画 | DW/PW 后 SiLU（可关） | v11 默认；纯 Conv+BN 无激活训练困难 |

两处开关都在 `DSNRblock.__init__` 参数里（`use_nam` / `use_act`），可自行实验对比。

## 7. 动手实验建议

```bash
# ① 看参数量对比
python tests/test_modules.py            # test_dsnrblock_lighter_than_standard_conv

# ② 把 use_nam 关掉，重新自检，观察参数量变化
#    （修改 tests/test_modules.py 或写个两行脚本对比 use_nam=True/False）

# ③ 训练后在 checkpoint 里查看每个 block 的门控值：
#    torch.load('runs/.../best.pt')['model'].model[2].nam_gate.sigmoid()
#    门控值>0.5 说明该层更依赖 NAM 注意力 —— 这是很有意思的可解释性证据
```

## 8. 本章自测

1. DSNRblock 的主支路两步卷积各自负责什么？（空间滤波 / 通道混合）
2. 为什么 NAM 能做到零新增参数？
3. 通道数不变的 DSNRblock，残差支路是什么？为什么？
4. 门控 g=0 和 g=1 时，模块分别退化成什么？

---

*下一篇 → [05-核心模块2：GhostConv 与 BiFPN 精讲](05-核心模块2-GhostConv与BiFPN精讲.md)*
