"""Reproducible, inference-only public infrared benchmark; no training or tuning."""
from pathlib import Path
import json, hashlib, csv, sys, time
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/2026-09-19'
RAW = REPO / 'datasets/sirst-v2'
import os
ORIGINAL_DATA = Path(os.environ['UVA_ORIGINAL_DATA']) if os.environ.get('UVA_ORIGINAL_DATA') else None
DEVICE = os.environ.get('UVA_DEVICE', '0' if torch.cuda.is_available() else 'cpu')
MODELS = {'UVA_cos': REPO/'models/UVA/best.pt',
          'UVA_linear': REPO/'models/UVA_no_cosine/best.pt'}

def read(p):
    p = Path(p)
    p = p if p.is_absolute() else REPO/p
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)

def digest(a):
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()

def write_csv(path, rows):
    if rows:
        with open(path,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def ious(a,b):
    a=np.asarray(a).reshape(-1,4); b=np.asarray(b).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh.prod(2)
    return inter/np.maximum(1e-12,(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter)

def match(gt,pred,conf=.25,iou=.5):
    p=np.asarray([p for p in pred if p[4]>=conf],float).reshape(-1,5)
    p=p[np.argsort(-p[:,4],kind='stable')]
    mat=ious(gt,p[:,:4]); used=set(); pairs=[]
    for j in range(len(p)):
        candidates=[k for k in range(len(gt)) if k not in used and mat[k,j]>=iou]
        if candidates:
            k=max(candidates,key=lambda k:mat[k,j]); used.add(k); pairs.append((k,j))
    return p,pairs,mat

def prepare():
    old=REPO/'datasets/sirst-v1'
    v1=set((old/'idx_427/trainvaltest.txt').read_text().split())
    test1=set((old/'idx_427/test.txt').read_text().split())
    test2=set((RAW/'splits/test_full.txt').read_text().split())
    entries=[]; hashes={}; mismatch=[]
    for p in sorted((RAW/'mixed').glob('*.png')):
        im=read(p); h,w=im.shape
        mask=read(RAW/'annotations/masks'/(p.stem+'_pixels0.png'))
        assert mask.shape==im.shape
        n,lab,stats,centers=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
        gt=[]; feats=[]
        for k,(x,y,bw,bh,area) in enumerate(stats[1:],1):
            gt.append([int(x),int(y),int(x+bw),int(y+bh)])
            pad=max(10,int(max(bw,bh))); x0=max(0,x-pad); x1=min(w,x+bw+pad); y0=max(0,y-pad); y1=min(h,y+bh+pad)
            crop=im[y0:y1,x0:x1].astype(float); bg=mask[y0:y1,x0:x1]==0
            values=crop[bg]; target=im[lab==k].astype(float)
            std=float(values.std()) if len(values) else 0.; mean=float(values.mean()) if len(values) else 0.
            gx=cv2.Sobel(crop,cv2.CV_64F,1,0); gy=cv2.Sobel(crop,cv2.CV_64F,0,1)
            feats.append(dict(gt_index=k-1,width=int(bw),height=int(bh),bbox_area=int(bw*bh),mask_area=int(area),
                              local_bg_std=std,local_bg_gradient=float(np.hypot(gx,gy)[bg].mean()) if bg.any() else 0.,
                              signed_contrast=float(target.mean()-mean),scr=float(abs(target.mean()-mean)/(std+1e-6))))
        dh=digest(im); hashes.setdefault(dh,[]).append(p.stem)
        entries.append(dict(id=p.stem,path=p.relative_to(REPO).as_posix(),width=w,height=h,gt=gt,features=feats,hash=dh,
                            v1=p.stem in v1,test_v1=p.stem in test1,test_v2=p.stem in test2))
    original=ORIGINAL_DATA/'images' if ORIGINAL_DATA else None
    overlaps=[]
    for split in (['train','val'] if original else []):
        for p in (original/split).glob('*'):
            if p.suffix.lower() not in ['.png','.jpg','.jpeg','.bmp']: continue
            dh=digest(read(p))
            if dh in hashes: overlaps.append(dict(original=str(p),public=hashes[dh],split=split))
    audit=dict(images=len(entries),targets=sum(len(e['gt']) for e in entries),backgrounds=sum(not e['gt'] for e in entries),
               old_images=len(v1),old_pixel_mismatches=None,exact_original_overlaps=overlaps if original else None,
               original_overlap_check_performed=original is not None,
               duplicate_groups=[v for v in hashes.values() if len(v)>1],
               annotation='8-connected components of official binary masks, tight zero-based half-open boxes; no component filtering',
               caveat='Exact decoded grayscale pixel overlap check only; cannot exclude same-sequence or transformed duplicates')
    (ROOT/'manifest.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('AUDIT',json.dumps(audit,ensure_ascii=False),flush=True)
    return entries

def summarize(entries,preds,name):
    subsets={'SIRST_v1_all':lambda e:e['v1'], 'SIRST_v1_official_test':lambda e:e['test_v1'],
             'SIRST_v2_full':lambda e:True,'SIRST_v2_official_test':lambda e:e['test_v2'],
             'SIRST_v2_new_targets':lambda e:not e['v1'] and bool(e['gt']),
             'SIRST_v2_backgrounds':lambda e:not e['gt']}
    rows=[]; targets=[]; images=[]
    for subset,fn in subsets.items():
        es=[e for e in entries if fn(e)]
        for thresh in [.05,.1,.25,.5]:
            tp=fp=fnn=fp_images=0
            for e in es:
                p,pairs,_=match(e['gt'],preds[e['id']],thresh)
                tp+=len(pairs); fp+=len(p)-len(pairs); fnn+=len(e['gt'])-len(pairs); fp_images+=len(p)>len(pairs)
            rows.append(dict(model=name,subset=subset,conf=thresh,iou=.5,images=len(es),gt=tp+fnn,tp=tp,fp=fp,fn=fnn,
                             precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fnn) if tp+fnn else None,
                             f1=2*tp/(2*tp+fp+fnn) if 2*tp+fp+fnn else None,fp_per_image=fp/len(es),images_with_fp=fp_images))
    for e in entries:
        p,pairs,mat=match(e['gt'],preds[e['id']]); matched={k for k,j in pairs}
        low,_,lowmat=match(e['gt'],preds[e['id']],.001)
        images.append(dict(model=name,id=e['id'],v1=e['v1'],gt=len(e['gt']),tp=len(pairs),fp=len(p)-len(pairs),fn=len(e['gt'])-len(pairs)))
        for k,feat in enumerate(e['features']):
            max_iou=float(mat[k].max()) if mat.shape[1] else 0.
            low_iou=float(lowmat[k].max()) if lowmat.shape[1] else 0.
            reason='detected' if k in matched else ('confidence_below_0.25' if low_iou>=.5 else ('localization_iou_0.1_to_0.5' if max_iou>=.1 else 'no_nearby_box'))
            targets.append(dict(model=name,id=e['id'],v1=e['v1'],detected=int(k in matched),max_iou=max_iou,max_iou_low_conf=low_iou,reason=reason,**feat))
    return rows,targets,images

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    force='--infer' in sys.argv
    entries=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8')) if (ROOT/'manifest.json').exists() else prepare()
    env=dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
             imgsz=1024,batch=8,rect=False,conf_floor=.001,match_iou=.5,main_conf=.25,
             weights={n:dict(path=p.relative_to(REPO).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in MODELS.items()})
    if force or not (ROOT/'environment.json').exists():
        (ROOT/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
    allrows=[]; allt=[]; alli=[]
    for name,path in MODELS.items():
        cache=ROOT/(name+'_predictions.json')
        if cache.exists() and not force: preds=json.loads(cache.read_text())
        else:
            model=YOLO(str(path)); preds={}; t=time.time()
            for start in range(0,len(entries),8):
                batch=entries[start:start+8]
                results=list(model.predict([str(REPO/e['path']) for e in batch],imgsz=1024,batch=8,rect=False,conf=.001,iou=.7,max_det=300,device=DEVICE,half=False,stream=True,verbose=False,save=False))
                assert len(results)==len(batch)
                for e,r in zip(batch,results):
                    assert tuple(r.orig_shape)==(e['height'],e['width'])
                    preds[e['id']]=r.boxes.data[:,:5].cpu().tolist()
                if start%104==0: print(name,start,'/',len(entries),'seconds',round(time.time()-t,1),flush=True)
            assert len(preds)==len(entries)
            cache.write_text(json.dumps(preds),encoding='utf-8')
        rows,ts,ims=summarize(entries,preds,name); allrows+=rows; allt+=ts; alli+=ims
        print(name,[r for r in rows if r['conf']==.25],flush=True)
    write_csv(ROOT/'metrics.csv',allrows); write_csv(ROOT/'target_details.csv',allt); write_csv(ROOT/'image_details.csv',alli)
    (ROOT/'metrics.json').write_text(json.dumps(allrows,indent=2),encoding='utf-8')

if __name__=='__main__': main()
