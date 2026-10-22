from pathlib import Path
import json,csv,math,hashlib
from html.parser import HTMLParser
import numpy as np
from PIL import Image
import markdown
ROOT=Path(r'C:\Users\Kyson\Desktop\检测结果')
DATASETS=['NUAA-SIRST','NUDT-SIRST']
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def pct(x):return f'{x*100:.2f}%'
def writecsv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
def main():
 metrics={};selections={};checks={};summary=[];excluded=[]
 for ds in DATASETS:
  d=ROOT/'results'/ds;m=load(d/'metrics.json');sel=load(d/'selection.json');gt=load(d/'ground_truth.json');manifest=load(d/'manifest.json');pred=load(d/'predictions_bbox.json')
  ids={im['id']:im for im in gt['images']};rows=list(csv.DictReader((d/'per_image.csv').open(encoding='utf-8-sig',newline='')))
  assert load(d/'progress.json')['status']=='complete'
  assert len(ids)==len(gt['images'])==len(manifest)==len(rows)==m['images']==sel['evaluated_images']
  assert len({r['path'] for r in manifest})==m['images']
  assert m['images']+len(sel['excluded'])==sel['test_images']
  assert m['box_padding_pixels_per_side']==0
  assert m['TP']+m['FN']==m['targets']==len(gt['annotations'])
  for k in ['tp','fp','fn']:assert sum(int(r[k]) for r in rows)==m[k.upper()]
  assert sum(p['score']>=.25 for p in pred)==m['TP']+m['FP']
  for p in pred:
   im=ids[p['image_id']];x,y,w,h=p['bbox'];assert all(math.isfinite(z) for z in [x,y,w,h,p['score']]);assert w>=0 and h>=0 and x>=-.001 and y>=-.001 and x+w<=im['width']+.001 and y+h<=im['height']+.001
  assert len(list((d/'visualizations').glob('*.png')))==m['images']
  for im in gt['images']:
   p=d/'predicted_masks'/f"{im['id']}.png"
   with Image.open(p) as img:assert img.size==(im['width'],im['height']);assert set(np.unique(np.asarray(img)))<={0,255}
  # Save convenient YOLO prediction txt files at the fixed operating point.
  grouped={i:[] for i in ids}
  for p in pred:
   if p['score']>=.25:grouped[p['image_id']].append(p)
  textdir=d/'prediction_txt';textdir.mkdir(exist_ok=True)
  for r in manifest:
   im=ids[r['id']];ls=[]
   for p in grouped[r['id']]:
    x,y,w,h=p['bbox'];ls.append(f"0 {(x+w/2)/im['width']:.8f} {(y+h/2)/im['height']:.8f} {w/im['width']:.8f} {h/im['height']:.8f} {p['score']:.8f}")
   (textdir/(Path(r['path']).stem+'.txt')).write_text('\n'.join(ls)+('\n' if ls else ''),encoding='utf-8')
  sheet=Image.open(d/'cases.jpg');sheet.crop((0,0,sheet.width,min(1280,sheet.height))).save(ROOT/f'{ds}_案例.jpg')
  metrics[ds]=m;selections[ds]=sel
  excluded.extend(dict(dataset=ds,**e) for e in sel['excluded'])
  checks[ds]=dict(status='passed',images=m['images'],targets=m['targets'],same_named_test_ids_in_model_train_validation=sel['test_filenames_in_model_train_val'],all_visualizations_present=True,all_binary_masks_shape_checked=True,box_padding=0)
  summary.append(dict(dataset=ds,images=m['images'],targets=m['targets'],box_AP50=m['bbox_coco']['AP50'],box_AP50_95=m['bbox_coco']['AP50_95'],precision=m['precision'],recall=m['recall'],F1=m['F1'],TP=m['TP'],FP=m['FP'],FN=m['FN'],mask_AP50=m['segmentation_coco']['AP50'],mask_AP50_95=m['segmentation_coco']['AP50_95'],pixel_IoU=m['pixel_IoU'],pixel_Dice=m['pixel_Dice']))
 for name,obj in [('指标.json',metrics),('数据选取说明.json',selections),('核验结果.json',checks),('排除样本.json',excluded)]:
  (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
 writecsv(ROOT/'指标汇总.csv',summary)
 total=sum(m['images'] for m in metrics.values())
 lines=['# NUAA-SIRST 与 NUDT-SIRST 模型检测报告','','日期：2026-10-20。沿用前一轮的 YOLO26n 分割最佳权重，仅执行推理与评估，未重新训练。','',
 f"从本地两套数据的 test 清单选取全部有效配对，共 **{total} 张图**。NUAA-SIRST 清单214张，其中1张图像/掩膜尺寸不一致被排除，实际213张；NUDT-SIRST 664张全部纳入。两套预测框均保持原始大小（pad=0）。",'',
 '## 1. 使用哪个模型','',
 '- 权重：`C:/Users/Kyson/Desktop/掩膜/all4_yolo26n_segment_best.pt`，单类别 target。',
 '- 用户表格中的 ACM、DNANet 用于说明 SIRST、NUDT-SIRST 的论文/数据集对应关系；此次调用的是上文已训练的 YOLO26n 分割模型。',
 '- NUAA-SIRST 和 NUDT-SIRST 都是此模型的训练数据来源。本次按已有 test 清单评估，属于同来源留出集测试，不属于上一轮 IRDST/ITSDT-15K 的新数据域测试。','',
 '## 2. 数据选取和质量检查','',
 '|数据集|清单图片数|实际检测|排除|文件名与现有模型测试划分一致|测试文件名与训练/验证交集|','|---|---:|---:|---:|---|---:|']
 for ds,s in selections.items():lines.append(f"|{ds}|{s['test_images']}|{s['evaluated_images']}|{len(s['excluded'])}|是|{s['test_filenames_in_model_train_val']}|")
 lines+=['','测试清单分别位于本地数据集的 `img_idx/test_NUAA-SIRST.txt` 与 `img_idx/test_NUDT-SIRST.txt`。文件名无交集不等于彻底排除同源场景或变换后的关联；本次不把它当作完全独立跨域数据。','',
 '**排除样本：NUAA-SIRST / Misc_111。** 原图325×220，掩膜592×400，缺少可信的空间对应关系。未强行缩放掩膜或修改原数据；这张图不进入准确性统计。排除依据是配对质量，不是模型分数。详细路径见 `排除样本.json`。','',
 '## 3. 指标口径','',
 '- 原始二值掩膜中非零像素为前景，以8连通域定义实例；框为每个实例的紧外接框，右下边界采用 xmax+1/ymax+1。没有轮廓多边形近似或目标筛除。',
 '- 输入 imgsz=1024、batch=8、GPU=0、retina_masks=True。保持预测框原始大小，不外扩。',
 '- 保存 conf≥0.001 的候选，用 pycocotools 的 COCO 101点插值计算 AP50 和 AP50–95；COCO maxDets=100，推理最多300个实例/图。',
 '- Precision/Recall/F1、TP/FP/FN 固定在 conf≥0.25、框 IoU≥0.50，以置信度降序一对一匹配。',
 '- mask AP 使用实例掩膜；像素 IoU、Dice 对工作点预测掩膜并集计算。没有以背景像素占比很高的普通像素准确率充当检测质量。',
 '- 本次使用原始像素掩膜；既有训练日志的验证器使用转换后的 YOLO 多边形标签，NUAA-SIRST 还少了上述无效样本，因此本次数值不应与旧日志直接当作同口径复现比较。','',
 '## 4. 框检测结果','',
 '|数据集|图像|目标框|AP50|AP50–95|Precision|Recall|F1|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for ds,m in metrics.items():lines.append(f"|{ds}|{m['images']}|{m['targets']}|{pct(m['bbox_coco']['AP50'])}|{pct(m['bbox_coco']['AP50_95'])}|{pct(m['precision'])}|{pct(m['recall'])}|{pct(m['F1'])}|")
 lines+=['','|数据集|TP 正确检测|FP 误检|FN 漏检|','|---|---:|---:|---:|']
 for ds,m in metrics.items():lines.append(f"|{ds}|{m['TP']}|{m['FP']}|{m['FN']}|")
 lines+=['','## 5. 分割结果','',
 '|数据集|mask AP50|mask AP50–95|全局像素 IoU|全局像素 Dice|像素 Precision|像素 Recall|','|---|---:|---:|---:|---:|---:|---:|']
 for ds,m in metrics.items():lines.append(f"|{ds}|{pct(m['segmentation_coco']['AP50'])}|{pct(m['segmentation_coco']['AP50_95'])}|{pct(m['pixel_IoU'])}|{pct(m['pixel_Dice'])}|{pct(m['pixel_precision'])}|{pct(m['pixel_recall'])}|")
 lines+=['','## 6. 图片结果','',
 '每张图均已保存可视化：绿色是真值框，红色是 conf≥0.25 的预测框，蓝色半透明区域为预测掩膜。预测掩膜 PNG 仅含0和255。','',
 '案例拼图前8张按工作点误检+漏检数量排序，后8张按测试清单均匀间隔抽取；拼图用于诊断，不代表随机抽样的总体表现。']
 for ds in DATASETS:lines+=['',f'### {ds}','',f'![{ds} 案例]({ds}_案例.jpg)','',f'[全部逐图可视化](results/{ds}/visualizations/) · [逐图指标](results/{ds}/per_image.csv) · [预测框 TXT](results/{ds}/prediction_txt/) · [完整案例图](results/{ds}/cases.jpg)']
 lines+=['','## 7. 保存文件和复现','',
 '- `指标汇总.csv`：可用 Excel 打开的汇总，指标以0–1小数保存；本报告按百分比显示。',
 '- `指标.json`：完整指标、模型 SHA256、设备、版本和运行参数。',
 '- `数据选取说明.json`、`排除样本.json`、`核验结果.json`：划分、排除理由与结果一致性核验。',
 '- `results/<数据集>/predictions_bbox.json`：COCO格式像素坐标框、分数，单类 category_id=1。',
 '- `prediction_txt/<原图名>.txt`：固定工作点的 YOLO格式 `class xc yc w h conf`，class=0、坐标归一化。',
 '- `predicted_masks/<image_id>.png`：预测掩膜并集；`manifest.json` 将 image_id 对应到原图路径。',
 '- 模型 SHA256：`'+next(iter(metrics.values()))['model_sha256']+'`。','',
 '复现命令：','```powershell',
 r'& "C:\Users\Kyson\anaconda3\envs\yolo26\python.exe" -X utf8 -u "C:\Users\Kyson\Desktop\检测结果\scripts\run_detection.py" NUAA-SIRST --box-pad 0',
 r'& "C:\Users\Kyson\anaconda3\envs\yolo26\python.exe" -X utf8 -u "C:\Users\Kyson\Desktop\检测结果\scripts\run_detection.py" NUDT-SIRST --box-pad 0',
 '```']
 text='\n'.join(lines);(ROOT/'检测报告.md').write_text(text,encoding='utf-8')
 body=markdown.markdown(text,extensions=['tables','fenced_code'])
 (ROOT/'检测报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>NUAA-SIRST 与 NUDT-SIRST 检测结果</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font-family:Segoe UI,Microsoft YaHei,sans-serif;line-height:1.8;color:#172033}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}td,th{border:1px solid #cbd5e1;padding:9px}th{background:#edf2f8}img{max-width:100%}pre{background:#f1f4f8;padding:16px;overflow:auto}a{color:#1766a1}</style>'+body+'</html>',encoding='utf-8')
 print('REPORT SAVED',ROOT,flush=True)
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
