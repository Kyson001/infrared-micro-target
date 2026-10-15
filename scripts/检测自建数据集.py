r"""用 掩膜 里的 YOLO26n 模型检测自建数据集 infarad micro-target，结果输出到 掩膜\自建集检测\。

说明：待测模型是分割模型（YOLO26n-seg），而自建数据集只有检测框标注、没有掩膜真值，
所以这里只评估"框"，指标自己算（贪心一对一匹配 + 全点插值 AP）：
    P / R / F1 @ conf=0.25, IoU=0.5；AP50；AP50-95（IoU 0.50:0.05:0.95 取平均）
模型只在四套公开红外数据上训练过，自建数据对它完全是新域，train/val 两个划分都算跨域测试。

输出：掩膜\自建集检测\
    预测txt\{train,val}\*.txt     YOLO 格式预测（cls xc yc w h conf）
    predictions.json              全部预测框
    可视化\*.png                   真值(绿) vs 预测(红) 对比图
    检测报告.json / 检测报告.md    指标与统计
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


MODEL = Path(r"C:\Users\Kyson\Desktop\掩膜\all4_yolo26n_segment_best.pt")
DATA_ROOT = Path(r"C:\Users\Kyson\Desktop\infarad micro-target\target data")
OUT = Path(r"C:\Users\Kyson\Desktop\掩膜\自建集检测")
IMGSZ = 1024
CONF = 0.25
N_VIS = 80


def read_gray(p: Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def write_img(p: Path, img) -> None:
    ok, buf = cv2.imencode(p.suffix, img)
    assert ok
    buf.tofile(str(p))


def load_gt(split: str):
    """读 YOLO 检测标签 -> {stem: [[x1,y1,x2,y2], ...]}（像素坐标）"""
    gt = {}
    for lab in sorted((DATA_ROOT / "labels" / split).glob("*.txt")):
        img_path = DATA_ROOT / "images" / split / (lab.stem + ".png")
        if not img_path.is_file():
            continue
        im = read_gray(img_path)
        if im is None:
            continue
        h, w = im.shape
        boxes = []
        for line in lab.read_text(encoding="utf-8").splitlines():
            q = line.split()
            if len(q) < 5:
                continue
            xc, yc, bw, bh = (float(v) for v in q[1:5])
            boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
        gt[lab.stem] = boxes
    return gt


def iou_matrix(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)))
    g = np.asarray(gt, float).reshape(-1, 4)
    p = np.asarray(pred, float).reshape(-1, 4)
    wh = np.maximum(0, np.minimum(g[:, None, 2:], p[None, :, 2:]) - np.maximum(g[:, None, :2], p[None, :, :2]))
    inter = wh.prod(2)
    ga = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    pa = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    return inter / np.maximum(1e-12, ga[:, None] + pa[None, :] - inter)


def ap_at_iou(gt_all, pred_all, iou_thr):
    """全点插值 AP（单类，COCO 口径）。"""
    records = []
    n_gt = sum(len(v) for v in gt_all.values())
    if n_gt == 0:
        return 0.0, 0, 0, 0
    for stem, gt in gt_all.items():
        preds = sorted(pred_all.get(stem, []), key=lambda q: -q[4])
        used = set()
        for q in preds:
            ious = iou_matrix(gt, [q[:4]])[:, 0] if len(gt) else np.zeros(0)
            best_i, best_v = -1, iou_thr
            for k in range(len(gt)):
                if k not in used and ious[k] >= best_v:
                    best_i, best_v = k, ious[k]
            if best_i >= 0:
                used.add(best_i)
                records.append((q[4], 1))
            else:
                records.append((q[4], 0))
    records.sort(key=lambda r: -r[0])
    tp = fp = 0
    prec, rec = [], []
    for _, is_tp in records:
        tp += is_tp
        fp += 1 - is_tp
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    # 全点插值
    mrec = [0.0] + rec + [rec[-1] if rec else 0.0]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    # conf=0.25 工作点
    sel = [r for r in records if r[0] >= CONF]
    tp25 = sum(r[1] for r in sel)
    fp25 = len(sel) - tp25
    fn25 = n_gt - tp25
    return ap * 100, tp25, fp25, fn25


def main() -> None:
    assert MODEL.is_file(), MODEL
    if OUT.is_dir():
        shutil.rmtree(str(OUT))
    (OUT / "预测txt").mkdir(parents=True)
    (OUT / "可视化").mkdir(parents=True)
    model = YOLO(str(MODEL))
    report = {"模型": str(MODEL), "模型大小MB": round(MODEL.stat().st_size / 1e6, 2),
              "数据集": str(DATA_ROOT), "imgsz": IMGSZ, "conf工作点": CONF,
              "时间": time.strftime("%Y-%m-%d %H:%M:%S"), "指标": {}}

    all_preds: dict = {}
    counts: dict = {}
    for split in ("train", "val"):
        (OUT / "预测txt" / split).mkdir(parents=True, exist_ok=True)
        imgs = sorted((DATA_ROOT / "images" / split).glob("*.png"))
        for i in range(0, len(imgs), 16):
            batch = imgs[i:i + 16]
            results = model.predict([str(p) for p in batch], imgsz=IMGSZ, batch=8, conf=0.05,
                                    iou=0.7, max_det=300, device=0, verbose=False, stream=True)
            for p, r in zip(batch, results):
                h, w = r.orig_shape
                lines, boxes = [], []
                for row in r.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, cf, cls = row[:6]
                    lines.append("{:.0f} {:.6f} {:.6f} {:.6f} {:.6f} {:.3f}".format(
                        cls, ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h, cf))
                    boxes.append([float(x1), float(y1), float(x2), float(y2), float(cf)])
                (OUT / "预测txt" / split / (p.stem + ".txt")).write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                key = split + "/" + p.stem
                all_preds[key] = boxes
                counts[key] = sum(1 for b in boxes if b[4] >= CONF)
        print("[推理] {} 完成：{} 张".format(split, len(imgs)), flush=True)
    (OUT / "predictions.json").write_text(json.dumps(all_preds, ensure_ascii=False), encoding="utf-8")

    # 指标
    for split in ("train", "val"):
        gt = load_gt(split)
        preds = {k.split("/", 1)[1]: v for k, v in all_preds.items() if k.startswith(split + "/")}
        ap50, tp, fp, fn = ap_at_iou(gt, preds, 0.50)
        aps = [ap_at_iou(gt, preds, t)[0] for t in np.arange(0.50, 1.0, 0.05)]
        n_gt = sum(len(v) for v in gt.values())
        report["指标"][split] = {
            "图像数": len(gt), "真值目标数": n_gt,
            "TP": tp, "FP": fp, "FN": fn,
            "P": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "R": round(tp / n_gt, 4) if n_gt else 0.0,
            "AP50": round(ap50, 4),
            "AP50_95": round(float(np.mean(aps)), 4),
            "每图预测框": round(sum(counts[k] for k in counts if k.startswith(split + "/")) / max(1, len(gt)), 3),
        }
        print("[指标] {split}: AP50={a:.4f} AP50-95={b:.4f} P={c:.4f} R={d:.4f} (TP={tp} FP={fp} FN={fn})".format(
            split=split, a=report["指标"][split]["AP50"], b=report["指标"][split]["AP50_95"],
            c=report["指标"][split]["P"], d=report["指标"][split]["R"], tp=tp, fp=fp, fn=fn), flush=True)

    # 可视化
    val_imgs = sorted((DATA_ROOT / "images" / "val").glob("*.png"))
    ranked = sorted(val_imgs, key=lambda p: -counts.get("val/" + p.stem, 0))
    random.seed(42)
    picks = ranked[: N_VIS // 2] + random.sample(val_imgs, min(N_VIS // 2, len(val_imgs)))
    gt_val = load_gt("val")
    for p in picks:
        img = read_gray(p)
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in gt_val.get(p.stem, []):
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 1)
        for x1, y1, x2, y2, cf in all_preds.get("val/" + p.stem, []):
            if cf < CONF:
                continue
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            cv2.putText(canvas, "{:.2f}".format(cf), (int(x1), max(10, int(y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        scale = max(1, int(600 / max(canvas.shape[:2])))
        vis = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        bar = np.full((26, vis.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, "{}  green=GT({})  red=pred({})".format(
            p.stem, len(gt_val.get(p.stem, [])), counts.get("val/" + p.stem, 0)),
            (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        write_img(OUT / "可视化" / (p.stem + ".png"), np.vstack([bar, vis]))
    print("[可视化] {} 张".format(len(picks)), flush=True)

    (OUT / "检测报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 自建数据集检测报告", "",
             "- 模型：`{}`（{:.2f} MB）".format(MODEL.name, report["模型大小MB"]),
             "- 数据集：`{}`（train {} 张 / val {} 张）".format(
                 DATA_ROOT, report["指标"]["train"]["图像数"], report["指标"]["val"]["图像数"]),
             "- 推理设置：imgsz={}, conf 工作点={}, IoU 匹配阈值=0.5".format(IMGSZ, CONF),
             "- 注意：模型只在四套**公开红外数据**上训练过，自建数据集是全新域，以下均为跨域结果。", "",
             "| 划分 | 图像 | 真值目标 | TP | FP | FN | P | R | AP50 | AP50-95 | 每图预测框 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split in ("train", "val"):
        s = report["指标"][split]
        lines.append("| {} | {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
            split, s["图像数"], s["真值目标数"], s["TP"], s["FP"], s["FN"], s["P"], s["R"], s["AP50"],
            s["AP50_95"], s["每图预测框"]))
    (OUT / "检测报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
