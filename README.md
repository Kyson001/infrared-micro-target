# infrared-micro-target
《基于华为昇腾AI处理器的红外微小目标处理器》

本仓库保存两份 YOLO26N best 模型、调用与评估代码、2026-09-19 实际使用的公开红外数据，以及完整预测、指标、漏检/虚警案例和中文报告。

**2026-10-15 更新：新增四数据集合并训练的分割模型 `all4_seg_4datasets`，并用它检测自建数据集 `infarad micro-target`。**
实测结果为 0 召回（跨域失败），原因排查与改进方向见
[自建集检测报告](results/2026-10-15_selfbuilt_eval/检测报告.md)、
[模型说明](models/all4_seg_4datasets/README.md) 与
[训练数据说明](datasets/README.md#四数据集合并训练集all42026-10-13)。

## 查看结果

- [完整评测报告与案例](results/2026-09-19/评测报告.md)
- [指标 CSV](results/2026-09-19/metrics.csv)
- [两模型案例总览](results/2026-09-19/case_contact_sheet.jpg)
- [四数据集模型：自建集检测报告（2026-10-15）](results/2026-10-15_selfbuilt_eval/检测报告.md)
- [四数据集模型：训练与分数据集指标（2026-10-15）](results/2026-10-15_all4_training/)
- [数据来源与实例定义](datasets/README.md)
- [模型说明](models/README.md)

## 四数据集模型（all4_seg_4datasets）

YOLO26n-seg、单类，合并 NUAA-SIRST + NUDT-SIRST + IRSTD-1K + MDFA 共 9,918 张训练图，100 轮（2026-10-13 完成）。
合并测试集：Box mAP50 **0.742**、Mask mAP50 **0.619**；逐数据集：

| 数据集 | NUDT-SIRST | IRSTD-1K | NUAA-SIRST | MDFA |
| --- | ---: | ---: | ---: | ---: |
| Box mAP50 | 0.896 | 0.746 | 0.569 | 0.208 |
| Mask mAP50 | 0.668 | 0.568 | 0.678 | 0.241 |

用该权重检测自建数据集 `infarad micro-target`（490×600、弥散低对比 UVA 目标）时，
conf≥0.25、IoU≥0.5 下 **TP = 0、AP50 = 0**；降到 conf=0.01、换分辨率、做对比度增强均无改善，
而同一批图用自建数据训练的模型可 20/20 检出。结论：**公开 IRSTD 域与自建 UVA 域存在系统性分布差异，
公开数据不能替代自建数据训练**，详见检测报告。

同一流程固定 `conf≥0.25`、`IoU≥0.5`，P 指检测精确率：

| 数据 | 模型 | P | R |
|---|---|---:|---:|
| 原验证集 1428 张 | UVA | 90.614% | 75.808% |
| 原验证集 1428 张 | 无余弦退火 | 91.411% | 74.916% |
| SIRST v2 全量 1024 张 | UVA | 0.077% | 0.309% |
| SIRST v2 全量 1024 张 | 无余弦退火 | 0.127% | 0.463% |

**这是跨数据域、统一 Target 类和掩膜紧框的压力测试，不是纯无人机业务准确率。** 原模型在原验证集正常，但公开数据上框尺度失配和背景虚警明显；完整报告同时提供官方 test 划分结果、中心定位诊断、阈值敏感性和局限。v1 与 v2 有重合图，不能当成独立数据叠加。

## 环境

本次实测：Python 3.12、PyTorch 2.5.1+cu121、Ultralytics 8.4.51、RTX 4060 Laptop 8 GB。CPU 可运行，但耗时更长。先建立独立环境，再安装依赖：

```bash
git clone https://github.com/Kyson001/infrared-micro-target.git
cd infrared-micro-target
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

无 NVIDIA CUDA 环境时，第一条安装命令使用 PyTorch 的 CPU 索引 `https://download.pytorch.org/whl/cpu`。代码默认检测 CUDA，否则用 CPU；可通过环境变量 `UVA_DEVICE` 选择评估设备。

## 调用任一模型

```bash
python scripts/predict.py --model UVA --source datasets/sirst-v2/mixed/Misc_1.png
python scripts/predict.py --model UVA_no_cosine --source datasets/sirst-v2/mixed/Misc_1.png
# --source 也支持自己的图片、文件夹或视频；--device cpu 可强制 CPU。
```

预测可视化和带置信度的 YOLO txt 输出到 `results/inference/`。

## 重新运行本次公开集评测

```bash
# 实际重新调用两份 best.pt，覆盖本地公开集预测和对应指标：
python scripts/evaluate.py --infer
# 生成逐目标/背景分析和案例图：
python scripts/analyze.py
# 生成 IoU/中心匹配诊断（未提供原始自建数据时自动跳过原集复核）：
python scripts/verify.py
# 生成中文报告：
python scripts/make_report.py
```

只想核对已发布预测的统计结果，可运行 `python scripts/evaluate.py`（不带 `--infer`，使用随仓库保存的预测）。主评估每批显式传入 8 张图，采用 1024 正方形 letterbox；匹配为按预测置信度排序的一对一 IoU 匹配。掩膜转框、特征计算和完整路径清单见代码与 `manifest.json`。

原训练/验证图像不属于本次公开数据上传范围；其原生验证、同流程固定阈值控制结果和逐图预测已保留。如需重跑，先设置 `UVA_ORIGINAL_DATA` 指向原数据集目录（含 `images/val`、`labels/val`、`dataset.yaml`），并确保 YAML 中的 path 适合当前设备：

```powershell
$env:UVA_ORIGINAL_DATA = 'C:\your-data\target-data'
python scripts/verify.py
python scripts/control.py
```

## 文件布局

```text
models/                     两份 best.pt、训练参数和训练指标
datasets/sirst-v2/           实际图像、掩膜、原生 XML 和划分
datasets/sirst-v1/           v1 划分；图像与 v2 共用，不重复存储
scripts/                    推理、评估、诊断、控制实验、报告生成
results/2026-09-19/          完整结果、预测 JSON、哈希、报告及案例图
```

权重是普通 Git 二进制文件，每份约 5.5 MB，不需要 Git LFS。数据逐文件保存，没有超过 GitHub 单文件限制的大压缩包。数据版权与引用见 [数据说明](datasets/README.md)，不对第三方数据另行授予许可。
