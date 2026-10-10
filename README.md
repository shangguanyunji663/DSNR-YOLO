# DSNR-YOLO 学习项目 —— 轻量化遥感目标检测（论文复现 + 从零学习）

> 复现论文：**《用于遥感目标检测的轻量化注意力残差增强 YOLO 架构》**
> （李晓帅等，地球信息科学学报 2026, 28(4):1174-1185），
> 原文 PDF 在 [`paper/`](paper/) 目录。
>
> 本项目 = **10 篇零基础学习文档** + **可运行的完整复现代码** + **自动化验证**。

---

## 🚀 快速开始（三条命令）

```bash
conda activate ML2                                   # 本机专用环境（勿用 base）
python tests/test_modules.py                         # 8 个模块单测（~10 秒）
python scripts/verify_model.py                       # 整网自检：构建/参数量/损失反传
```

想看它真的能训练？再来一条（~1 分钟，GPU）：

```bash
python scripts/make_toy_dataset.py
python scripts/train.py --data data/toy.yaml --epochs 3 --imgsz 320 --batch 4 --nbs 4 --device 0
```

## 📚 学习从这里开始

**两条入口任选**：
- **[docs/13 从零到精通完全课程表](docs/13-从零到精通-完全课程表.md)** —— 18 课时完整课程：每课时的必会术语、核心问题（附参考答案）、动手实验、毕业标准；配套 [docs/14 专业术语手册](docs/14-专业术语手册.md)（约 120 词三段式讲解）。**零基础推荐从这里开始。**
- **[docs/00 学习路线图](docs/00-学习路线图.md)** —— 六阶段路线 + 文件地图（已具备基础者用）。

| 文档 | 内容 |
|---|---|
| [13 完全课程表](docs/13-从零到精通-完全课程表.md) | ⭐ 18 课时从零到出师 + 全部答案 |
| [14 术语手册](docs/14-专业术语手册.md) | ⭐ 约 120 个术语的完整词典 |
| [00 学习路线图](docs/00-学习路线图.md) | 六阶段计划 + 文件地图 |
| [01 前置知识A](docs/01-前置知识A-目标检测与YOLO.md) | 检测是什么、IoU/mAP 手算、YOLO 思想 |
| [02 前置知识B](docs/02-前置知识B-卷积参数量与轻量化技术.md) | 参数量公式、DepSep/Ghost 算例、注意力家族、FPN→BiFPN |
| [03 论文精读](docs/03-论文精读-背景动机与整体架构.md) | 问题定义、技术选型、图1 逐层解读 |
| [04 DSNRblock 精讲](docs/04-核心模块1-DSNRblock逐行精讲.md) | 论文核心模块：式(1)-(7) × 图2 × 代码三对照 |
| [05 GhostConv 与 BiFPN](docs/05-核心模块2-GhostConv与BiFPN精讲.md) | 幽灵卷积 + 双向加权金字塔 |
| [06 混合损失精讲](docs/06-核心模块3-自适应混合损失函数精讲.md) | 式(8)-(11)：ATFL+SLIDE+可学习权重 |
| [07 实验解读](docs/07-实验设计-数据集指标与结果解读.md) | 表1~表6 逐格讲解 |
| [08 代码导读](docs/08-代码导读-项目结构与ultralytics注册原理.md) | 张量流追踪 + 三大注册补丁原理 |
| [09 复现指南](docs/09-复现指南-环境数据训练与调参.md) | 环境坑、数据准备、正式训练、排查表 |
| [10 拓展阅读](docs/10-拓展阅读-日报两篇论文导读.md) | 日报两篇论文导读：DROS（参数高效微调）与 YOLO12-MambaScan（同题不同器） |
| [11 精读教程①DROS](docs/11-精读教程1-DROS参数高效微调.md) | LoRA/低秩/SVD/子空间从零讲起，逐组件拆解"轻训练"路线 |
| [12 精读教程②MambaScan](docs/12-精读教程2-YOLO12-MambaScan.md) | 图像频率/空洞卷积/坐标注意力/Mamba 从零讲起，五模块对照精读 |

## 🔬 论文的四大改进（复现状态）

| 改进 | 效果 | 代码 | 状态 |
|---|---|---|---|
| ① DSNRblock（主干） | 9.43M→8.00M 参数，精度保持 | `dsnr_yolo/modules/dsnr_block.py` | ✅ 已实现+测试 |
| ② GhostConv（下采样/颈部） | 8.00M→6.69M，16.1 GFLOPs | ultralytics 内置，yaml 直接用 | ✅ 已接入 |
| ③ BiFPN（颈部） | 精度反超基线（88.8 vs 88.4） | `dsnr_yolo/modules/bifpn.py` + yaml | ✅ 已实现+测试 |
| ④ ATFL+SLIDE 混合损失 | 再 +0.2 mAP50 | `dsnr_yolo/losses/mixed_loss.py` | ✅ 已实现+反传验证 |

本复现整网实测：**6.01M 参数 / ≈17 GFLOPs / 三尺度 P3-P5**
（论文 7.15M/18.3，差异来自论文未公开的颈部细节，见 docs/05 的诚实声明）。

## 🖥️ 环境说明

- **conda `ML2` 环境**（`D:\Anaconda\envs\ML2`，Python 3.12.12）
  - torch 2.11.0+cu128（GPU: RTX 4060 Laptop 8GB ✓）
  - ultralytics 8.4.164
- ⚠️ **不使用、不修改 base 环境**
- 重装/换机步骤见 [docs/09 第 1 节](docs/09-复现指南-环境数据训练与调参.md)。

## ✅ 已验证清单

- [x] 模块单元测试 8/8 通过（`tests/test_modules.py`）
- [x] 整网自检通过（`scripts/verify_model.py`：结构/参数量/三尺度形状/损失反传）
- [x] 玩具数据集 GPU 冒烟训练通过（`runs/`），混合损失可学习权重
      ω₁/ω₂ 确认被优化器更新（1.0 → 0.9969）
- [ ] DIOR 正式复现训练（待你下载数据集，见 `data/README.md` 与 docs/09）

## 📁 目录

```
├── README.md            ← 本文件
├── requirements.txt
├── paper/               ← 论文原文
├── docs/                ← 10 篇学习文档
├── dsnr_yolo/           ← 复现代码包（modules / losses / models / 注册补丁）
├── scripts/             ← 自检 / 玩具数据 / 数据转换 / 训练 / 评估
├── tests/               ← 模块单元测试
└── data/                ← 数据集说明 + 配置模板（toy 自动生成）
```

## 📌 引用

```
李晓帅, 胡成玺, 田子建, 等. 用于遥感目标检测的轻量化注意力残差增强YOLO架构[J].
地球信息科学学报, 2026, 28(4): 1174-1185.
DOI: 10.12082/dqxxkx.2026.250477
```
