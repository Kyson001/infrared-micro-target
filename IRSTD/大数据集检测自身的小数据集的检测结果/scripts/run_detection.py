from pathlib import Path
import argparse,json,time,csv,hashlib,platform,re
from collections import defaultdict
import numpy as np
import cv2,torch,ultralytics
from PIL import Image,ImageDraw
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskutil

OUT=Path(r'C:\Users\Kyson\Desktop\检测结果')
WEIGHT=Path(r'C:\Users\Kyson\Desktop\掩膜\all4_yolo26n_segment_best.pt')
def pad_boxes_xyxy(boxes,shape,pad=2):
 """Expand boxes in original-image pixels; preserve input boxes and mask geometry."""
 if not np.isfinite(pad) or pad<0:raise ValueError('box padding must be finite and nonnegative')
 h,w=shape
 boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4).copy()
 boxes[:,0]=np.clip(boxes[:,0]-pad,0,w)
 boxes[:,1]=np.clip(boxes[:,1]-pad,0,h)
 boxes[:,2]=np.clip(boxes[:,2]+pad,0,w)
 boxes[:,3]=np.clip(boxes[:,3]+pad,0,h)
 return boxes
def read(p):
 a=cv2.imdecode(np.fromfile(str(p),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
 if a is None:raise ValueError(f'Cannot decode {p}')
 return a
def save(p,a):
 p.parent.mkdir(parents=True,exist_ok=True);ok,b=cv2.imencode(p.suffix,a);assert ok;b.tofile(str(p))
def rle(a):
 r=maskutil.encode(np.asfortranarray(a.astype(np.uint8)));r['counts']=r['counts'].decode();return r
def iou(a,b):
 a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
 inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
 return inter/np.maximum(1e-12,(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter)
def fixed(gt,pred):
 used=set();tp=0
 for p in sorted(pred,key=lambda x:-x['score']):
  x,y,w,h=p['bbox'];v=iou(gt,[[x,y,x+w,y+h]])[:,0];c=[i for i in np.argsort(-v) if i not in used and v[i]>=.5]
  if c: used.add(c[0]);tp+=1
 return tp,len(pred)-tp,len(gt)-tp
def coco_eval(gt,preds,typ):
 c=COCO();c.dataset=gt;c.createIndex()
 if preds:d=c.loadRes(preds)
 else:
  d=COCO();d.dataset={'images':gt['images'],'categories':gt['categories'],'annotations':[]};d.createIndex()
 e=COCOeval(c,d,typ);e.params.catIds=[1];e.evaluate();e.accumulate();e.summarize()
 return dict(AP50_95=float(e.stats[0]),AP50=float(e.stats[1]),AP75=float(e.stats[2]),AR100=float(e.stats[8]))

def prepare(name,dest=None,limit=0):
 dest=dest if dest is not None else OUT/'results'/name
 root=Path(r'C:\Users\Kyson\Desktop\分割检测')/name
 split_file=root/'img_idx'/('test_'+name+'.txt')
 stems=[s.strip() for s in split_file.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
 assert len(stems)==len(set(stems)), 'duplicate entries in test split'
 merged=Path(r'C:\Users\Kyson\Desktop\分割检测\yolo_dataset\all4_segment\images')
 test_stems={p.stem.removeprefix(name+'__') for p in (merged/'test').glob(name+'__*.png')}
 train_val_stems={p.stem.removeprefix(name+'__') for split in ['train','val'] for p in (merged/split).glob(name+'__*.png')}
 assert set(stems)==test_stems, 'test split differs from recorded model test set'
 assert not set(stems)&train_val_stems, 'test filenames overlap model training or validation'
 selection=dict(dataset=name,source=str(root),split_file=str(split_file),selected_split='test',test_images=len(stems),test_names_match_model_test=True,test_filenames_in_model_train_val=0,
   evaluation_scope='Held-out test split from a dataset used as part of this model training sources; not unseen-domain evaluation.',
   ground_truth_definition='Original nonzero binary mask, 8-connected components as instances; half-open tight boxes [xmin,ymin,xmax+1,ymax+1]. No polygon approximation.')
 (dest/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
 if limit:stems=stems[:limit]
 images=[];anns=[];records=[];aid=1;excluded=[]
 for stem in stems:
  p=root/'images'/(stem+'.png');mp=root/'masks'/(stem+'.png')
  a=read(p);mask=read(mp)>0
  if mask.shape!=a.shape:
   excluded.append(dict(stem=stem,reason='image_mask_dimension_mismatch',image=str(p),mask=str(mp),image_shape=list(a.shape),mask_shape=list(mask.shape)));continue
  i=len(records)+1
  h,w=a.shape;images.append(dict(id=i,width=w,height=h,file_name=str(p)))
  n,lab,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
  for k in range(1,n):
   x,y,bw,bh,area=map(int,stats[k]);anns.append(dict(id=aid,image_id=i,category_id=1,bbox=[x,y,bw,bh],area=area,segmentation=rle(lab==k),iscrowd=0));aid+=1
  records.append(dict(id=i,path=str(p),mask=str(mp),sequence='single_frame_test'))
 selection.update(evaluated_images=len(records),excluded=excluded)
 (dest/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
 return dict(info={'description':name+' original-mask test evaluation'},images=images,annotations=anns,categories=[dict(id=1,name='target')]),records

def main():
 ap=argparse.ArgumentParser();ap.add_argument('dataset',choices=['NUAA-SIRST','NUDT-SIRST']);ap.add_argument('--limit',type=int,default=0)
 ap.add_argument('--box-pad',type=float,default=None,help='Outward padding per side in original-image pixels; default=0 for these two datasets. Use 0 for original boxes.')
 ap.add_argument('--output-dir',type=Path,default=None,help='Separate result directory for a new evaluation run.')
 args=ap.parse_args();name=args.dataset
 pad=args.box_pad if args.box_pad is not None else 0.0
 if not np.isfinite(pad) or pad<0:ap.error('--box-pad must be finite and nonnegative')
 if args.limit<0:ap.error('--limit must be nonnegative')
 run_name=name+(f'_pad{pad:g}' if pad else '')+(f'_limit{args.limit}' if args.limit else '')
 dest=args.output_dir if args.output_dir is not None else OUT/'results'/run_name
 dest.mkdir(parents=True,exist_ok=True)
 gt,records=prepare(name,dest=dest,limit=args.limit)
 if args.limit:
  records=records[:args.limit];ids={r['id'] for r in records};gt['images']=[x for x in gt['images'] if x['id'] in ids];gt['annotations']=[x for x in gt['annotations'] if x['image_id'] in ids]
 (dest/'ground_truth.json').write_text(json.dumps(gt),encoding='utf-8')
 (dest/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 byid=defaultdict(list)
 for a in gt['annotations']:
  x,y,w,h=a['bbox'];byid[a['image_id']].append([x,y,x+w,y+h])
 model=YOLO(str(WEIGHT));print('MODEL',model.task,model.names,'IMAGES',len(records),'GT',len(gt['annotations']),'BOX PAD',pad,'OUTPUT',dest,flush=True)
 preds=[];rawpreds=[];spreds=[];rows=[];timings=[];start=time.time();total_i=total_u=total_p=total_g=total_pixels=0
 for off in range(0,len(records),8):
  batch=records[off:off+8]
  outputs=model.predict([r['path'] for r in batch],imgsz=1024,conf=.001,iou=.7,max_det=300,device=0,retina_masks=True,verbose=False,save=False)
  for rec,res in zip(batch,outputs):
   h,w=res.orig_shape;local=[];union=np.zeros((h,w),dtype=np.uint8)
   rawbs=res.boxes.xyxy.cpu().numpy()
   bs=pad_boxes_xyxy(rawbs,res.orig_shape,pad)
   scores=res.boxes.conf.cpu().numpy();masks=res.masks.data.cpu().numpy() if res.masks is not None else []
   for j,(box,score) in enumerate(zip(bs,scores)):
    if pad:
     rx1,ry1,rx2,ry2=map(float,rawbs[j]);rawpreds.append(dict(image_id=rec['id'],category_id=1,bbox=[rx1,ry1,rx2-rx1,ry2-ry1],score=float(score)))
    x1,y1,x2,y2=map(float,box);p=dict(image_id=rec['id'],category_id=1,bbox=[x1,y1,x2-x1,y2-y1],score=float(score));preds.append(p);local.append(p)
    if j<len(masks):
     mm=masks[j]>0.5
     assert mm.shape==(h,w),(mm.shape,(h,w))
     if name!='ITSDT-15K':spreds.append(dict(image_id=rec['id'],category_id=1,segmentation=rle(mm),score=float(score)))
     if score>=.25:union|=mm.astype(np.uint8)
   sel=[p for p in local if p['score']>=.25];tp,fp,fn=fixed(byid[rec['id']],sel)
   row=dict(id=rec['id'],path=rec['path'],sequence=rec['sequence'],gt=len(byid[rec['id']]),pred=len(sel),tp=tp,fp=fp,fn=fn)
   if name!='ITSDT-15K':
    m=read(Path(rec['mask']))>0;inter=int(np.count_nonzero(m & (union>0)));u=int(np.count_nonzero(m | (union>0)));npix=int(union.sum());ng=int(m.sum())
    row.update(pixel_intersection=inter,pixel_union=u,predicted_pixels=npix,gt_pixels=ng,pixel_iou=inter/u if u else 1.)
    total_i+=inter;total_u+=u;total_p+=npix;total_g+=ng;total_pixels+=h*w
   save(dest/'predicted_masks'/f"{rec['id']}.png",union*255)
   visual=res.orig_img.copy()
   if union.any():
    overlay=visual.copy();overlay[union>0]=(255,100,0);visual=cv2.addWeighted(visual,.55,overlay,.45,0)
   for x1,y1,x2,y2 in byid[rec['id']]:cv2.rectangle(visual,(int(x1),int(y1)),(int(x2-1),int(y2-1)),(0,255,0),1)
   for q in sel:
    x,y,bw,bh=q['bbox'];cv2.rectangle(visual,(int(x),int(y)),(int(x+bw),int(y+bh)),(0,0,255),1)
   cv2.putText(visual,'GT green | Pred red | Mask blue | conf 0.25',(5,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
   save(dest/'visualizations'/f"{rec['id']:04d}_{Path(rec['path']).stem}.png",visual)
   timings.append(res.speed);rows.append(row)
  if off%80==0 or off+8>=len(records):print(f'{name}: {min(off+8,len(records))}/{len(records)}, {time.time()-start:.1f}s',flush=True)
  # Partial predictions retained so a failed run is inspectable.
  if off%400==0:(dest/'progress.json').write_text(json.dumps({'completed':len(rows),'total':len(records),'elapsed_seconds':time.time()-start}),encoding='utf-8')
 (dest/'predictions_bbox.json').write_text(json.dumps(preds),encoding='utf-8')
 if pad:(dest/'predictions_bbox_unpadded.json').write_text(json.dumps(rawpreds),encoding='utf-8')
 if name!='ITSDT-15K':(dest/'predictions_segmentation.json').write_text(json.dumps(spreds),encoding='utf-8')
 with (dest/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
 summary=dict(dataset=name,images=len(rows),targets=len(gt['annotations']),model=str(WEIGHT),model_sha256=hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),imgsz=1024,prediction_conf=.001,operating_conf=.25,operating_box_iou=.5,nms_iou=.7,max_det=300,TP=tp,FP=fp,FN=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),F1=2*tp/max(1,2*tp+fp+fn),bbox_coco=coco_eval(gt,preds,'bbox'),inference_wall_seconds=time.time()-start,ultralytics=ultralytics.__version__,torch=torch.__version__,GPU=torch.cuda.get_device_name(0),speed_ms={k:float(np.mean([t[k] for t in timings])) for k in timings[0]})
 summary['end2end']=bool(getattr(model.model,'end2end',False));summary['nms_applied']=not summary['end2end']
 summary['box_padding_pixels_per_side']=pad
 summary['box_padding_coordinate_space']='original_image_pixels'
 summary['output_directory']=str(dest)
 summary['postprocessing_note']='Only predicted box coordinates are padded and clipped. Mask geometry, ground truth, and confidence scores are unchanged. Padded-box results are a separate postprocessing experiment.'
 if name!='ITSDT-15K':
  maskgt=json.loads((dest/'ground_truth_mask.json').read_text()) if name=='IRDST' else gt
  if args.limit:
   ids={r['id'] for r in records};maskgt['images']=[x for x in maskgt['images'] if x['id'] in ids];maskgt['annotations']=[x for x in maskgt['annotations'] if x['image_id'] in ids]
  summary.update(mask_targets=len(maskgt['annotations']),segmentation_coco=coco_eval(maskgt,spreds,'segm'),pixel_IoU=total_i/max(1,total_u),pixel_Dice=2*total_i/max(1,total_p+total_g),mean_image_IoU=float(np.mean([r['pixel_iou'] for r in rows])),pixel_precision=total_i/max(1,total_p),pixel_recall=total_i/max(1,total_g),false_positive_pixels_per_million=(total_p-total_i)/max(1,total_pixels)*1e6)
 seq=[]
 for s in sorted({r['sequence'] for r in rows}):
  rs=[r for r in rows if r['sequence']==s];t=sum(r['tp'] for r in rs);f=sum(r['fp'] for r in rs);n=sum(r['fn'] for r in rs)
  seq.append(dict(sequence=s,images=len(rs),targets=sum(r['gt'] for r in rs),TP=t,FP=f,FN=n,precision=t/max(1,t+f),recall=t/max(1,t+n),F1=2*t/max(1,2*t+f+n)))
 summary['sequences']=seq
 (dest/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 # Deterministic difficult and typical cases, with full view plus local zoom.
 chosen=sorted(rows,key=lambda r:-(r['fn']+r['fp']))[:8]+rows[::max(1,len(rows)//8)][:8]
 sheet=Image.new('RGB',(1400,len(chosen)*320),'#202020');dd=ImageDraw.Draw(sheet)
 for k,row in enumerate(chosen):
  im=Image.open(row['path']).convert('RGB');d=ImageDraw.Draw(im)
  boxes=byid[row['id']];pp=[p for p in preds if p['image_id']==row['id'] and p['score']>=.25]
  for b in boxes:d.rectangle(tuple(b),outline='#00ff50',width=2)
  for p in pp:
   x,y,w,h=p['bbox'];d.rectangle((x,y,x+w,y+h),outline='#ff3030',width=2)
  im.thumbnail((470,285));sheet.paste(im,(0,k*320+25));dd.text((4,k*320+5),f"{name} {Path(row['path']).parent.name}/{Path(row['path']).name} GT:{row['gt']} TP:{row['tp']} FP:{row['fp']} FN:{row['fn']}",fill='white')
  original=Image.open(row['path']).convert('RGB');ds=ImageDraw.Draw(original)
  for b in boxes:ds.rectangle(tuple(b),outline='#00ff50',width=1)
  for p in pp:
   x,y,w,h=p['bbox'];ds.rectangle((x,y,x+w,y+h),outline='#ff3030',width=1)
  if boxes:
   b=boxes[0];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2;rad=max(24,min(120,max(b[2]-b[0],b[3]-b[1])*2));crop=original.crop((int(cx-rad),int(cy-rad),int(cx+rad),int(cy+rad))).resize((280,280),Image.Resampling.NEAREST);sheet.paste(crop,(490,k*320+25))
  mask=Image.open(dest/'predicted_masks'/f"{row['id']}.png").convert('RGB');mask.thumbnail((460,280));sheet.paste(mask,(810,k*320+25));dd.text((815,k*320+5),'Predicted mask @ conf 0.25',fill='white')
 sheet.save(dest/'cases.jpg');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
 (dest/'progress.json').write_text(json.dumps({'completed':len(rows),'total':len(records),'status':'complete'}),encoding='utf-8')
if __name__=='__main__':main()
