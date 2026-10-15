# all4_seg_4datasets：四套公开红外数据合并训练的分割模型

YOLO26n-seg、单类 `target`，用 **NUAA-SIRST + NUDT-SIRST + IRSTD-1K + MDFA** 四套公开数据
合并训练 100 轮得到。这是目前本仓库中"公开数据覆盖面最广"的一份权重，可作为迁移学习的初始化。

## 训练配置

| 项 | 值 |
| --- | --- |
| 模型 | `yolo26n-seg.pt`（COCO 预训练）→ 100 轮微调 |
| 任务 | 实例分割（同时输出框与掩膜） |
| 训练数据 | 合并后 train 9,918 / val 1,748 / test 1,179 张（划分文件见 `../../datasets/train_splits_4datasets/`） |
| 数据构成 | MDFA 8,492 + IRSTD-1K 680 + NUDT-SIRST 564 + NUAA-SIRST 182（train 部分） |
| 超参数 | `imgsz=1024`、`batch=8`、`epochs=100`、`optimizer=auto`、`cos_lr=True`、`seed=42`、`patience=30` |
| 实测耗时 | 2.32 小时（RTX 4060 Laptop 8 GB） |
| 完成时间 | 2026-10-13 |

## 测试集指标（Ultralytics `val`，合并测试集 1,179 张 / 1,505 个目标）

| 指标 | 值 |
| --- | ---: |
| Box mAP50 | 0.742 |
| Box mAP50-95 | 0.371 |
| Mask mAP50 | 0.619 |
| Mask mAP50-95 | 0.251 |

## 分数据集测试（同一权重，各自的官方测试划分）

| 数据集 | 测试图像 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NUDT-SIRST | 664 | 0.896 | 0.490 | 0.668 | 0.298 |
| IRSTD-1K | 201 | 0.746 | 0.325 | 0.568 | 0.190 |
| NUAA-SIRST | 214 | 0.569 | 0.227 | 0.678 | 0.236 |
| MDFA | 100 | 0.208 | 0.055 | 0.241 | 0.060 |

注：MDFA 的 1 万张训练图是**合成**（真实背景注入目标），而其 100 张测试图是真实红外图像，
因此该列偏低反映的是"合成→真实"的差距，不代表模型在 MDFA 训练分布上失效。

## 使用

```bash
# 对单张图/文件夹推理
python scripts/predict_selfbuilt.py   # 或直接用 Ultralytics：YOLO("best.pt").predict(source=...)

# 逐数据集复评
python scripts/按数据集评估.py
```

## 重要限制

该模型**不能**直接用于自建数据集 `infarad micro-target`（弥散、低对比的 UVA 目标）：
2026-10-15 的实测显示 conf≥0.25、IoU≥0.5 下召回为 0，详见
[`../../results/2026-10-15_selfbuilt_eval/检测报告.md`](../../results/2026-10-15_selfbuilt_eval/检测报告.md)。
若要用于自建域，请先在自建数据上微调。
