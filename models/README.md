# 训练权重

| 模型 | 文件 | 调度 |
|---|---|---|
| UVA | `UVA/best.pt` | `cos_lr=True`，余弦退火 |
| UVA（无余弦退火） | `UVA_no_cosine/best.pt` | `cos_lr=False`，线性衰减 |

均为 YOLO26N、单类 UVA、imgsz=1024、seed=42，100 轮训练得到的 best 权重。无余弦组曾中断后续训。两份权重按字节原样上传；SHA256 在 `../results/2026-09-19/environment.json` 中。

各文件夹同时保留训练 `args.yaml` 与 `results.csv`。args 中原设备的桌面路径替换成 `${ORIGINAL_DESKTOP}`，只用于训练配置溯源，不能直接当作跨设备可运行配置。权重内部保留原训练元数据。

本次评测只推理，未重新训练。此模型的 UVA 类别不意味着能够识别所有 SIRST Target。
