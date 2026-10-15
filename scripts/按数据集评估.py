r"""把"三套合并训练"得到的模型，分别在三套数据各自的官方测试划分上评估。

做法：从合并测试集（文件名带数据集前缀）用硬链接拆出三份子集，不额外占磁盘，
然后分别调用 YOLO 的 val，得到每个数据集的 box / mask 指标。

运行：
    & "C:\Users\Kyson\anaconda3\envs\yolo26\python.exe" "C:\Users\Kyson\Desktop\分割检测\按数据集评估.py"
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ultralytics import YOLO


MERGED = Path(r"C:\Users\Kyson\Desktop\分割检测\yolo_dataset\all4_segment")
OUT = Path(r"C:\Users\Kyson\Desktop\分割检测\yolo_dataset\by_dataset")
WEIGHTS = Path(r"C:\Users\Kyson\Desktop\掩膜\all4_yolo26n_segment_best.pt")
RESULT_JSON = Path(r"C:\Users\Kyson\Desktop\掩膜\分数据集结果.json")
NAMES = ["NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K", "MDFA"]


def prepare(name: str) -> Path:
    """用硬链接把某个数据集的测试图/标签拆出来，返回 data.yaml。"""
    for sub in ("images", "labels"):
        (OUT / name / sub / "test").mkdir(parents=True, exist_ok=True)
    for p in (MERGED / "images" / "test").glob(name + "__*.png"):
        dst = OUT / name / "images" / "test" / p.name
        if not dst.exists():
            try:
                os.link(p, dst)
            except OSError:
                import shutil
                shutil.copy2(str(p), str(dst))
    for p in (MERGED / "labels" / "test").glob(name + "__*.txt"):
        dst = OUT / name / "labels" / "test" / p.name
        if not dst.exists():
            try:
                os.link(p, dst)
            except OSError:
                import shutil
                shutil.copy2(str(p), str(dst))
    yaml_path = OUT / name / "data.yaml"
    yaml_path.write_text(
        "path: {}\ntrain: images/test\nval: images/test\ntest: images/test\n\nnames:\n  0: target\n".format(
            (OUT / name).as_posix()),
        encoding="utf-8")
    return yaml_path


def main() -> None:
    assert WEIGHTS.is_file(), WEIGHTS
    model = YOLO(str(WEIGHTS))
    results = {}
    for name in NAMES:
        yaml_path = prepare(name)
        n_img = len(list((OUT / name / "images" / "test").glob("*.png")))
        m = model.val(data=str(yaml_path), split="test", imgsz=1024, batch=8, device=0,
                      project=r"C:\Users\Kyson\Desktop\掩膜", name="分数据集评估_" + name)
        results[name] = {
            "images": n_img,
            "box_mAP50": float(m.box.map50), "box_mAP50_95": float(m.box.map),
            "mask_mAP50": float(m.seg.map50), "mask_mAP50_95": float(m.seg.map),
        }
        print("[{}] 图像={} box mAP50={:.4f} mAP50-95={:.4f} | mask mAP50={:.4f} mAP50-95={:.4f}".format(
            name, n_img, results[name]["box_mAP50"], results[name]["box_mAP50_95"],
            results[name]["mask_mAP50"], results[name]["mask_mAP50_95"]), flush=True)

    # 合并测试集（训练脚本里已算过，可从 结果.json 读取）作为对照
    summary_json = Path(r"C:\Users\Kyson\Desktop\掩膜\all4_yolo26n_segment_结果.json")
    if summary_json.is_file():
        s = json.loads(summary_json.read_text(encoding="utf-8"))
        results["四套合并测试集"] = {
            "images": None,
            "box_mAP50": s.get("box_mAP50"), "box_mAP50_95": s.get("box_mAP50_95"),
            "mask_mAP50": s.get("mask_mAP50"), "mask_mAP50_95": s.get("mask_mAP50_95"),
        }
    RESULT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n结果已保存：", RESULT_JSON)


if __name__ == "__main__":
    main()
