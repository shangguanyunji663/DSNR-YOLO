# 数据集准备说明

本项目用到论文中的两个遥感数据集。**两者都需要手动下载**（有注册/网盘门槛），
下载后按下面的目录结构摆放，再用模板 yaml 指定路径。

## 1. DIOR（主实验数据集，论文表2/3/5）

- 全称：Detection in Optical Remote Sensing Images
- 规模：23 463 张图，190 000+ 实例，20 类（飞机/机场/体育场/储罐/风车/烟囱/车辆…）
- 官方论文：Li K, Wan G, Cheng G, et al. ISPRS 2020（参考文献[25]）
- 获取方式：在 GitHub/Gitee 搜索 "DIOR dataset"（常青课题组页面附官方申请/网盘链接），
  或通过论文 `paper/` 目录里参考文献 [25] 的 DOI 页面找来源。

原始标注是 **PASCAL VOC 风格的 XML**（水平框 xmin/ymin/xmax/ymax）。
论文 3.1.2 节说明其按 **8:1:1** 随机划分训练/验证/测试集 —— 本项目提供的
`scripts/convert_dior_to_yolo.py` 会同时完成 "XML → YOLO txt" 与 "8:1:1 划分"。

转换后的目录结构（ultralytics 标准格式）：

```
data/dior/
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/
```

用法：

```bash
# 假设原始数据在 D:/datasets/DIOR（内含 Annotations/ 与 JPEGImages-support/ 等子目录）
python scripts/convert_dior_to_yolo.py --src D:/datasets/DIOR --dst data/dior
```

## 2. SIMD（对比实验数据集，论文表4/6）

- 全称：Satellite Imagery Multi-vehicles Dataset
- 规模：5 000 张 RGB 图（1024×768），15 类（轿车/卡车/公交/飞机/船…）
- 官方论文：Haroon M, et al. IEEE JSTARS 2020（参考文献[24]）
- 获取方式：IEEE DataPort / Kaggle 搜索 "SIMD dataset satellite"

SIMD 的标注同样是 XML（VOC 风格），可复用同一个转换脚本：

```bash
python scripts/convert_dior_to_yolo.py --src D:/datasets/SIMD --dst data/simd
```

## 3. 数据集 yaml 模板

- `data/dior.yaml` —— 主实验（nc=20）
- `data/simd.yaml` —— 对比实验（nc=15）
- `data/toy.yaml` —— 自动生成的玩具数据集（冒烟测试用，勿用于评估）

把 yaml 里的 `path:` 改成你机器上的实际绝对路径即可训练：

```bash
python scripts/train.py --data data/dior.yaml --epochs 200 --imgsz 640 --batch 8 --device 0
```

## 4. 类别顺序注意

转换脚本按"字母序"固化类别顺序并写入 yaml 的 names。**训练与后续评估必须用同一份
yaml**，否则类别号错位、mAP 会错。若你用其它来源的 YOLO 格式数据，请核对
`data/*.yaml` 里的 names 顺序与 labels 类别 id 一一对应。
