r"""一键：解压 → 掩膜转 YOLO 标签 → GPU 训练 YOLO26n(seg/detect) → 测试集评估 → 导出模型。

【当前默认 = C 计划】四套数据合并训练（NUAA-SIRST + NUDT-SIRST + IRSTD-1K + MDFA 1 万张），
imgsz=1024、batch=8、100 轮、结果输出到 C:\Users\Kyson\Desktop\掩膜\all4_yolo26n_segment\。
直接用 PyCharm 运行本文件（解释器选 yolo26）就是 C 计划，不需要加任何参数。
预计耗时 16–20 小时（约 12 分钟/轮），建议睡前启动、别让电脑休眠。

其它常用组合：
    --dataset all              三套公开集（不含 MDFA），约 2.5 小时/100 轮
    --dataset MDFA --imgsz 512 MDFA 单独训练，约 5 小时
    --limit-mdfa 3000          只用 MDFA 的前 3000 张，用于快速对照实验
    --patience 0               关闭早停，强制跑满 100 轮

【重要】必须用装了 PyTorch / Ultralytics 的解释器运行，不要用 Python38：
    & "C:\Users\Kyson\anaconda3\envs\yolo26\python.exe" ^
      "C:\Users\Kyson\Desktop\分割检测\train_yolo26n.py" --dataset NUAA-SIRST

    或者直接双击同目录下的 run_train.bat（里面已经写死了正确解释器）。

常用参数：
    --dataset all,MDFA | all | NUAA-SIRST | NUDT-SIRST | IRSTD-1K | MDFA   要训练的数据集，默认 all,MDFA（四套合并）
    --task segment | detect                                 默认分割；detect 会用掩膜生成检测框
    --epochs 100 --batch 8 --imgsz 1024 --device 0
    --model <路径>                                          默认 yolo26n-seg.pt；传 yolo26n-seg.yaml 即从零训练
    --resume                                                从中断处继续（自动找 last.pt）
    --out-root <目录>                                        模型与训练产物的输出根目录（默认 桌面\掩膜）
    --data-root <目录>                                       转换后的 YOLO 数据存放目录（默认 分割检测\yolo_dataset）

产物（结果目录 out-root = C:\Users\Kyson\Desktop\掩膜）：
    转换后的数据   C:\Users\Kyson\Desktop\分割检测\yolo_dataset\all4_segment\   （数据放在数据目录，不进结果目录）
    训练过程/曲线  C:\Users\Kyson\Desktop\掩膜\all4_yolo26n_segment\
    模型文件       C:\Users\Kyson\Desktop\掩膜\all4_yolo26n_segment_best.pt / _last.pt
    评估结果       C:\Users\Kyson\Desktop\掩膜\all4_yolo26n_segment_结果.json
    预测可视化     C:\Users\Kyson\Desktop\掩膜\all4_yolo26n_segment_predict\
    文件说明       C:\Users\Kyson\Desktop\掩膜\结果说明.md
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path


# ------------------------------------------------------------------
# 0) 依赖自检：如果用错了解释器（例如 Python38），给出明确的中文提示
# ------------------------------------------------------------------
missing = []
try:
    import cv2
except Exception:
    missing.append("opencv-python")
try:
    import numpy as np
except Exception:
    missing.append("numpy")
try:
    import torch
except Exception:
    missing.append("torch")
try:
    from ultralytics import YOLO
except Exception:
    missing.append("ultralytics")

if missing:
    print("=" * 78)
    print("缺少运行依赖：", ", ".join(missing))
    print("当前使用的解释器：", sys.executable)
    print()
    print("这个脚本必须用装了 PyTorch + Ultralytics 的环境运行，请改用下面这个解释器：")
    print()
    print(r'    & "C:\Users\Kyson\anaconda3\envs\yolo26\python.exe" "' + str(Path(__file__).resolve()) + '"')
    print()
    print(r"或者直接双击同目录下的 run_train.bat（已指定正确解释器）。")
    print("=" * 78)
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path(r"C:\Users\Kyson\Desktop\掩膜")            # 训练结果（模型/曲线/指标/预测图）输出目录
DATA_ROOT = ROOT / "yolo_dataset"                         # 转换后的 YOLO 数据目录（留在数据盘里）
DATASETS = {
    "NUAA-SIRST": "train_NUAA-SIRST.txt",
    "NUDT-SIRST": "train_NUDT-SIRST.txt",
    "IRSTD-1K": "train_IRSTD-1K.txt",
    "MDFA": None,          # 1 万张合成数据，结构不同（data/training/NNNNNN_1.png + _2.png），单独处理
}
DEFAULT_SEG_WEIGHTS = Path(r"C:\Users\Kyson\Desktop\yolo26\yolo26n-seg.pt")
DEFAULT_DET_WEIGHTS = Path(r"C:\Users\Kyson\Desktop\yolo26\yolo26n.pt")


# ------------------------------------------------------------------
# 1) 数据准备：解压 + 掩膜转 YOLO 标签
# ------------------------------------------------------------------
def read_gray(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def ensure_extracted(name):
    """数据集没解压就自动解压。"""
    target = ROOT / name
    if (target / "images").is_dir():
        return target
    zip_path = ROOT / (name + ".zip")
    if not zip_path.is_file():
        raise FileNotFoundError("既没有解压目录也没有压缩包：{} / {}".format(target, zip_path))
    print("[解压] {} → {}".format(zip_path.name, target), flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    return target


def mask_to_polygons(mask):
    """掩膜 → 多边形列表（像素坐标）。1~2 像素的极小目标退化成点/线，这里兜底成小方框。"""
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        pts = c.reshape(-1, 2)
        if len(pts) < 3:
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0) + 1
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        polys.append(pts)
    return polys


def mask_to_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mdfa_items(split, limit=0):
    """MDFA（1 万张合成图）：data/training/NNNNNN_1.png(原图) + _2.png(掩膜)；测试集在 test_org/test_gt。"""
    base = ROOT / "MDFA" / "data"
    if not base.is_dir():
        raise FileNotFoundError(
            "找不到 MDFA 数据：{}\n请先用 7-Zip 解压分卷（示例）：\n"
            '  & "C:\\Users\\Kyson\\anaconda3\\envs\\tools\\bin\\7z.exe" x '
            '"C:\\Users\\Kyson\\Desktop\\分割检测\\downloads_MDFA\\data.part1.rar" '
            '-o"C:\\Users\\Kyson\\Desktop\\分割检测\\MDFA" -y'.format(base))
    items = []
    if split == "train":
        files = sorted((base / "training").glob("*_1.png"))
        if limit:
            files = files[:limit]
        for p in files:
            stem = p.stem[:-2]
            items.append((p, base / "training" / (stem + "_2.png"), stem))
    else:
        for p in sorted((base / "test_org").glob("*.png")):
            items.append((p, base / "test_gt" / p.name, p.stem))
    return items


def build_yolo_dataset(names, task, val_from_train, seed, force, data_root, limit_mdfa=0):
    tag = "_".join(names) if len(names) <= 2 else ("all4" if "MDFA" in names else "all")
    out = data_root / "{}_{}".format(tag, task)
    if out.is_dir() and not force and (out / "data.yaml").is_file():
        print("[数据] 已有转换结果，跳过转换：{}".format(out), flush=True)
        return out / "data.yaml"
    if out.is_dir() and force:
        shutil.rmtree(str(out))
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    stats = {}
    for name in names:
        # 先取"训练池"和"测试池"，再切验证集
        if name == "MDFA":
            pool = mdfa_items("train", limit_mdfa)
            test_items = mdfa_items("test")
        else:
            d = ensure_extracted(name)
            split_file = d / "img_idx" / DATASETS[name]
            test_file = d / "img_idx" / split_file.name.replace("train_", "test_")
            train_stems = [s.strip() for s in split_file.read_text(encoding="utf-8").split() if s.strip()]
            test_stems = [s.strip() for s in test_file.read_text(encoding="utf-8").split() if s.strip()]
            pool = [(d / "images" / (s + ".png"), d / "masks" / (s + ".png"), s) for s in train_stems]
            test_items = [(d / "images" / (s + ".png"), d / "masks" / (s + ".png"), s) for s in test_stems]

        random.Random(seed).shuffle(pool)
        n_val = max(1, int(len(pool) * val_from_train))
        val_items, train_items = pool[:n_val], pool[n_val:]
        stats[name] = dict(train=len(train_items), val=len(val_items), test=len(test_items))
        prefix = name + "__" if len(names) > 1 else ""
        skipped = 0
        for split, items in (("train", train_items), ("val", val_items), ("test", test_items)):
            for img_src, mask_src, stem in items:
                if not img_src.is_file() or not mask_src.is_file():
                    skipped += 1
                    continue
                mask = read_gray(mask_src)
                if mask is None:          # 少量图片本身损坏（MDFA 里有 22 张），跳过并计数
                    skipped += 1
                    continue
                stem_out = prefix + stem
                shutil.copy2(str(img_src), str(out / "images" / split / (stem_out + ".png")))
                h, w = mask.shape
                if task == "segment":
                    lines = []
                    for poly in mask_to_polygons(mask):
                        coords = []
                        for x, y in poly:
                            coords += [min(1.0, max(0.0, (float(x) + 0.5) / w)),
                                       min(1.0, max(0.0, (float(y) + 0.5) / h))]
                        if len(coords) >= 6:
                            lines.append("0 " + " ".join("{:.6f}".format(v) for v in coords))
                    text = "\n".join(lines) + ("\n" if lines else "")
                else:
                    box = mask_to_box(mask)
                    text = ""
                    if box:
                        x0, y0, x1, y1 = box
                        text = "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                            ((x0 + x1) / 2.0) / w, ((y0 + y1) / 2.0) / h,
                            (x1 - x0) / float(w), (y1 - y0) / float(h))
                (out / "labels" / split / (stem_out + ".txt")).write_text(text, encoding="utf-8")
        s = stats[name]
        print("[数据] {}: train={} val={} test={}{}".format(
            name, s["train"], s["val"], s["test"],
            "（跳过损坏/缺失 {} 张）".format(skipped) if skipped else ""), flush=True)

    (out / "data.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: target\n".format(out.as_posix()),
        encoding="utf-8")
    (out / "dataset_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return out / "data.yaml"


# ------------------------------------------------------------------
# 2) 训练 + 评估 + 导出
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all,MDFA",
                    help="默认 all,MDFA（C 计划：四套合并训练）；也可写 all（不含 MDFA）或单套名字，多套用逗号分隔")
    ap.add_argument("--task", default="segment", choices=("segment", "detect"))
    ap.add_argument("--model", default="", help="留空则 segment 用 yolo26n-seg.pt、detect 用 yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30,
                    help="早停耐心值：连续多少轮没有提升就提前结束；传 0 表示关闭早停、跑满 epochs")
    ap.add_argument("--batch", type=int, default=8, help="8 GB 显存建议 8；显存不足降到 4")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-from-train", type=float, default=0.15)
    ap.add_argument("--limit-mdfa", type=int, default=0,
                    help="MDFA 只用前 N 张训练图（0=全部 1 万张）；想快速试跑可以传 2000")
    ap.add_argument("--force-convert", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--predict", type=int, default=8, help="训练后在测试集上出多少张预测图")
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="训练结果输出目录（默认 桌面\\掩膜）")
    ap.add_argument("--data-root", default=str(DATA_ROOT), help="转换后的 YOLO 数据存放目录（默认 分割检测\\yolo_dataset）")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    (out_root / "结果说明.md").write_text(
        "# 训练结果目录说明\n\n"
        "这个文件夹里只有**训练结果**，转换后的数据在 `C:\\Users\\Kyson\\Desktop\\分割检测\\yolo_dataset\\`。\n\n"
        "## 每次训练会产生这些东西\n\n"
        "| 文件/目录 | 作用 |\n"
        "| --- | --- |\n"
        "| `<名字>\\weights\\best.pt` | **最佳权重**，评估、部署、写论文用这个 |\n"
        "| `<名字>\\weights\\last.pt` | 最后一轮权重，只用于 `--resume` 续训 |\n"
        "| `<名字>\\args.yaml` | 本次训练的全部超参数（复现用） |\n"
        "| `<名字>\\results.csv` | 每轮的 loss 与指标，**画曲线、做表格的数据源** |\n"
        "| `<名字>\\results.png` | 上面 csv 的可视化总览 |\n"
        "| `<名字>\\Box*_curve.png` | 检测框的 PR/P/R/F1 曲线 |\n"
        "| `<名字>\\Mask*_curve.png` | 掩膜的 PR/P/R/F1 曲线（分割任务才有） |\n"
        "| `<名字>\\confusion_matrix*.png` | 混淆矩阵 |\n"
        "| `<名字>\\labels.jpg` | 训练集标注分布（框大小、数量），用来查数据异常 |\n"
        "| `<名字>\\train_batch*.jpg` | 训练时的数据增强样例 |\n"
        "| `<名字>\\val_batch*_labels.jpg` / `val_batch*_pred.jpg` | **验证集真值 vs 预测对照**，最直观的质量检查 |\n"
        "| `<名字>_best.pt` / `<名字>_last.pt` | 从 weights 复制出来的权重，方便直接取用 |\n"
        "| `<名字>_结果.json` | 汇总：数据集、配置、box/mask 指标、实际训练轮数、完成时间 |\n"
        "| `<名字>_test\\` | 在**测试集**上重新评估产生的全套曲线 |\n"
        "| `<名字>_predict\\` | 测试集上的预测可视化图 |\n\n"
        "## 手上有这些就够用了\n\n"
        "- 要模型：`<名字>_best.pt`\n"
        "- 要数字：`<名字>_结果.json`、`<名字>\\results.csv`\n"
        "- 要图：`<名字>\\results.png`、`BoxPR_curve.png`、`MaskPR_curve.png`、`<名字>_predict\\`\n"
        "- 要复现：`<名字>\\args.yaml` + 转换后的数据\n",
        encoding="utf-8")

    # 环境
    if torch.cuda.is_available():
        print("[环境] CUDA 可用：{}，显存 {:.1f} GB".format(
            torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1024 ** 3), flush=True)
    else:
        print("[环境] 警告：CUDA 不可用，退回 CPU（会很慢）", flush=True)
        args.device = "cpu"

    # 数据
    # 解析 --dataset："all" 展开为三套公开集（不含 MDFA），可与其它名字用逗号组合，如 all,MDFA
    names = []
    for token in [t.strip() for t in args.dataset.split(",") if t.strip()]:
        if token.lower() == "all":
            names += ["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"]
        else:
            assert token in DATASETS, "未知数据集 {}，可选 {} 或 all".format(token, list(DATASETS))
            names.append(token)
    seen = set()
    names = [n for n in names if not (n in seen or seen.add(n))]
    assert names, "没有解析到任何数据集"
    data_yaml = build_yolo_dataset(names, args.task, args.val_from_train, args.seed, args.force_convert,
                                   data_root, args.limit_mdfa)
    print("[数据] data.yaml = {}".format(data_yaml), flush=True)
    ds_root = data_yaml.parent
    counts = {s: len(list((ds_root / "images" / s).glob("*.png"))) for s in ("train", "val", "test")}
    print("[数据] 合并后：train={train} val={val} test={test}（共 {} 张）".format(
        sum(counts.values()), **counts), flush=True)
    # 粗略预估耗时（按本机实测约 0.05 秒/张/轮@imgsz1024）
    per_img = 0.05 if args.imgsz >= 1024 else (0.025 if args.imgsz >= 640 else 0.012)
    est_h = (counts["train"] + counts["val"]) * per_img * args.epochs / 3600.0
    print("[计划] {} 轮 × {} 张/轮，预计约 {:.1f} 小时"
          "（若中途早停会更短；想跑满加 --patience 0）".format(
              args.epochs, counts["train"] + counts["val"], est_h), flush=True)

    tag = "_".join(names) if len(names) <= 2 else ("all4" if "MDFA" in names else "all")
    name = args.name or "{}_yolo26n_{}".format(tag, args.task)
    run_dir = out_root / name

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        assert last.is_file(), "找不到可续训的权重：{}".format(last)
        print("[训练] 从 {} 继续训练".format(last), flush=True)
        YOLO(str(last)).train(resume=True)
    else:
        weights = args.model or str(DEFAULT_SEG_WEIGHTS if args.task == "segment" else DEFAULT_DET_WEIGHTS)
        print("[训练] 模型={} epochs={} batch={} imgsz={} 数据={}".format(
            weights, args.epochs, args.batch, args.imgsz, data_yaml), flush=True)
        model = YOLO(weights)
        extra = {"overlap_mask": True, "mask_ratio": 4} if args.task == "segment" else {}
        model.train(data=str(data_yaml), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                    device=args.device, workers=args.workers, seed=args.seed, optimizer="auto",
                    patience=args.patience, cos_lr=True, rect=False, amp=True,
                    project=str(out_root), name=name, plots=True, **extra)
        print("[训练] 完成：{}".format(run_dir / "weights"), flush=True)

    # 实际训练轮数（早停时会小于 --epochs）
    csv_path = run_dir / "results.csv"
    actual_epochs = 0
    if csv_path.is_file():
        actual_epochs = max(0, len(csv_path.read_text(encoding="utf-8").strip().splitlines()) - 1)
    if actual_epochs and actual_epochs < args.epochs:
        print("[提示] 计划 {} 轮，实际只跑了 {} 轮：连续 {} 轮没有提升触发了早停（best.pt 仍是最佳权重的）。"
              "想跑满就把 --patience 调大，或传 --patience 0 关闭早停。".format(
                  args.epochs, actual_epochs, args.patience), flush=True)

    # 测试集评估
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    use = best if best.is_file() else last
    model = YOLO(str(use))
    metrics = model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch,
                        device=args.device, project=str(out_root), name=name + "_test")
    result = {
        "数据集": names, "任务": args.task, "权重来源": args.model or (
            str(DEFAULT_SEG_WEIGHTS) if args.task == "segment" else str(DEFAULT_DET_WEIGHTS)),
        "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz, "seed": args.seed,
        "patience": args.patience, "实际训练轮数": actual_epochs,
        "box_mAP50": float(metrics.box.map50), "box_mAP50_95": float(metrics.box.map),
        "mask_mAP50": float(metrics.seg.map50) if args.task == "segment" else None,
        "mask_mAP50_95": float(metrics.seg.map) if args.task == "segment" else None,
        "训练目录": str(run_dir), "完成时间": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("[测试集] box  mAP50={:.4f}  mAP50-95={:.4f}".format(result["box_mAP50"], result["box_mAP50_95"]))
    if args.task == "segment":
        print("[测试集] mask mAP50={:.4f}  mAP50-95={:.4f}".format(
            result["mask_mAP50"], result["mask_mAP50_95"]))

    # 导出模型到 out-root（C:\Users\Kyson\Desktop\掩膜）
    exported = []
    for src, suffix in ((best, "_best.pt"), (last, "_last.pt")):
        if src.is_file():
            dst = out_root / (name + suffix)
            shutil.copy2(str(src), str(dst))
            exported.append(str(dst))
    result["导出模型"] = exported
    (out_root / (name + "_结果.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[导出] 模型已放到：{}".format("、".join(exported)), flush=True)

    # 预测可视化
    if args.predict > 0:
        test_dir = Path(str(data_yaml).replace("data.yaml", "")) / "images" / "test"
        imgs = sorted(test_dir.glob("*.png"))
        random.Random(args.seed).shuffle(imgs)
        if imgs:
            model.predict([str(p) for p in imgs[: args.predict]], imgsz=args.imgsz, device=args.device,
                          conf=0.25, save=True, project=str(out_root), name=name + "_predict")
            print("[预测] 可视化图：{}".format(out_root / (name + "_predict")), flush=True)

    print("\n全部完成。\n  模型：{}\n  结果：{}".format(
        out_root / (name + "_best.pt"), out_root / (name + "_结果.json")))


if __name__ == "__main__":
    main()
