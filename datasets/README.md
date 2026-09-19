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
