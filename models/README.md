# 训练权重

| 模型 | 文件 | 训练数据 | 调度 |
|---|---|---|---|
| UVA | `UVA/best.pt` | 自建 `infarad micro-target`（单类 UVA） | `cos_lr=True`，余弦退火 |
| UVA（无余弦退火） | `UVA_no_cosine/best.pt` | 自建 `infarad micro-target`（单类 UVA） | `cos_lr=False`，线性衰减 |
| **all4_seg_4datasets** | `all4_seg_4datasets/best.pt` | **四套公开数据合并**（NUAA-SIRST + NUDT-SIRST + IRSTD-1K + MDFA） | `cos_lr=True`，YOLO26n-**seg** 分割模型 |

## 前两份（自建 UVA 域）

均为 YOLO26N、单类 UVA、imgsz=1024、seed=42，100 轮训练得到的 best 权重。无余弦组曾中断后续训。两份权重按字节原样上传；SHA256 在 `../results/2026-09-19/environment.json` 中。

各文件夹同时保留训练 `args.yaml` 与 `results.csv`。args 中原设备的桌面路径替换成 `${ORIGINAL_DESKTOP}`，只用于训练配置溯源，不能直接当作跨设备可运行配置。权重内部保留原训练元数据。

本次评测只推理，未重新训练。此模型的 UVA 类别不意味着能够识别所有 SIRST Target。

## 第三份（四套公开数据合并，2026-10-13）

YOLO26n-seg，单类 `target`，合并 NUAA-SIRST / NUDT-SIRST / IRSTD-1K / MDFA 共 9,918 张训练图，
100 轮、imgsz=1024、batch=8、seed=42，实测 2.32 小时（RTX 4060 Laptop）。
合并测试集（1,179 张）上 Box mAP50 = 0.742、Mask mAP50 = 0.619；逐数据集成绩与完整说明见
[`all4_seg_4datasets/README.md`](all4_seg_4datasets/README.md)。

**注意**：该权重在自建数据集上实测召回为 0（弥散低对比目标不在其训练分布内），
详见 [`../results/2026-10-15_selfbuilt_eval/检测报告.md`](../results/2026-10-15_selfbuilt_eval/检测报告.md)。
