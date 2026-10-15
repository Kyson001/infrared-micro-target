# 本次实际评测数据

| 来源 | 官方链接 | 本仓库内容 |
|---|---|---|
| SIRST v1 | https://github.com/YimianDai/sirst | 原始划分与上游 README；427 张图像与对应掩膜共用下方 v2 文件 |
| SIRST v2 | https://github.com/YimianDai/open-sirst-v2 | 1024 张 mixed 图像、1024 张二值掩膜、上游 XML 框/掩膜元数据和划分 |

2026-09-19 从两个官方仓库 master 下载。下载归档的 SHA256 见 `../results/2026-09-19/source_archives.json`；仓库内实际上传文件的 SHA256 见 `file_checksums.json`。

## 文件对应

- 图像：`sirst-v2/mixed/<id>.png`
- 掩膜：`sirst-v2/annotations/masks/<id>_pixels0.png`
- 原生框：`sirst-v2/annotations/bboxes/<id>.xml`
- v1 全量清单：`sirst-v1/idx_427/trainvaltest.txt`
- v1 官方测试：`sirst-v1/idx_427/test.txt`
- v2 官方测试：`sirst-v2/splits/test_full.txt`

两版本的 427 张重合图已核验像素完全一致，因此只存一次。v2 全量有 514 张含目标图及 510 张纯背景图；含目标图中 87 张为相对 v1 的新增图。

评测没有直接使用 XML 框，而是将掩膜的 8 连通域转成零基、右下角不包含的紧外接框，不过滤小连通域。按此定义，v1 有 533 个实例，v2 有 648 个；这不保证与其他论文的实例定义一致。

## 来源归属

数据和原始标注归原作者/权利人所有，本仓库不为第三方数据另行授予许可证。上游归档未附带独立 LICENSE 文件；不要把公开可下载解释为任意用途授权，具体使用与再分发条件以原作者说明为准。保留两份 `UPSTREAM_README.md` 和其中的论文引用。

本次没有上传用户原训练/验证图像；报告中的原验证结果及预测记录作为实测对照保留。IRSTD-1k、SIRST-UAVB 未在本次运行，不包含于本仓库。

---

# 四数据集合并训练集（all4，2026-10-13）

`models/all4_seg_4datasets/best.pt` 的训练数据来自以下四套公开数据集。**本仓库只保存划分清单与转换脚本，
不重复上传第三方原始图像**（合计约 400 MB，且原始数据版权归各作者，按官方入口下载更合适）。

| 数据集 | 官方入口 | 规模 | 标注 |
|---|---|---|---|
| NUAA-SIRST（SIRST v1） | https://github.com/YimianDai/sirst | 427 张（train 213 / test 214） | 二值掩膜 |
| NUDT-SIRST | https://github.com/YeRen123455/Infrared-Small-Target-Detection | 1327 张（train 663 / test 664） | 二值掩膜 |
| IRSTD-1K | https://github.com/RuiZhang97/ISNet | 1001 张（train 800 / test 201） | 二值掩膜 |
| MDFA（MDvsFA） | https://github.com/wanghuanphd/MDvsFA_cGAN | 10,000 对训练 + 100 对测试 | 二值掩膜（合成目标注入真实背景） |

## 实际使用的划分

- 三套 IRSTD 数据用各自官方 `img_idx/train_*.txt`、`test_*.txt`；
- MDFA 用 `data/training/*_1.png`（原图）与 `_2.png`（掩膜），测试集为 `data/test_org` + `data/test_gt`；
- 从训练池中按 seed=42 随机切出 15% 作为验证集；
- 合并后：**train 9,918 / val 1,748 / test 1,179**，逐文件清单见
  [`train_splits_4datasets/train.txt`](train_splits_4datasets/train.txt)、`val.txt`、`test.txt`，
  类别定义见 [`train_splits_4datasets/data.yaml`](train_splits_4datasets/data.yaml)。

文件名带数据集前缀（`MDFA__000001.png` 等），可直接还原来源。

## 复现转换与训练

```bash
# 1) 按上表下载四套数据，解压到 <工作目录>/<数据集名>/{images,masks,img_idx}
#    （MDFA 为分卷 RAR，用 7-Zip 解压到 <工作目录>/MDFA/data）
# 2) 转换 + 训练（脚本已含掩膜→YOLO 多边形标签、15% 验证集切分、EPOCH/批量等参数）
python scripts/train_yolo26n.py                 # 默认即四套合并、imgsz=1024、100 轮
# 3) 逐数据集复评
python scripts/按数据集评估.py
```

转换细节：掩膜取 `>0` 为前景，用 8 连通域外轮廓转归一化多边形（1–2 像素的极小目标兜底成最小矩形），
背景图写空标签文件。MDFA 原始包中有 22 张训练图损坏（PNG 解码失败），转换时自动跳过。

上述四套数据版权归原作者所有，本仓库不为其另行授予许可。
