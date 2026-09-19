"""Repeat identical public inference/matching on the original validation set."""
import json
from evaluate import *

def main():
    if ORIGINAL_DATA is None:
        raise SystemExit('Set UVA_ORIGINAL_DATA to the original dataset directory first.')
    data=ORIGINAL_DATA
    entries=[]; sizes=[]; duplicate_labels=0
    for p in sorted((data/'images/val').glob('*.png')):
        im=read(p); h,w=im.shape
        label=data/'labels/val'/(p.stem+'.txt')
        raw=[tuple(map(float,line.split())) for line in label.read_text().splitlines() if line.strip()]
        labels=list(dict.fromkeys(raw)); duplicate_labels+=len(raw)-len(labels); gt=[]
        for cls,x,y,bw,bh in labels:
            gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]); sizes.append([bw*w,bh*h])
        entries.append(dict(id=p.stem,path=str(p),width=w,height=h,gt=gt))
    rows=[]
    for name,path in MODELS.items():
        model=YOLO(str(path)); tp=fp=fn=0; preds={}
        for start in range(0,len(entries),8):
            batch=entries[start:start+8]
            rs=list(model.predict([e['path'] for e in batch],imgsz=1024,batch=8,rect=False,conf=.001,iou=.7,max_det=300,device=DEVICE,half=False,stream=True,verbose=False))
            assert len(rs)==len(batch)
            for e,r in zip(batch,rs):
                pred=r.boxes.data[:,:5].cpu().tolist(); preds[e['id']]=pred
                p,pairs,_=match(e['gt'],pred); tp+=len(pairs); fp+=len(p)-len(pairs); fn+=len(e['gt'])-len(pairs)
            if start%400==0: print(name,start,flush=True)
        (ROOT/(name+'_original_predictions.json')).write_text(json.dumps(preds),encoding='utf-8')
        rows.append(dict(model=name,images=len(entries),gt=tp+fn,conf=.25,iou=.5,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp),recall=tp/(tp+fn)))
        print(rows[-1],flush=True)
    result=dict(metrics=rows,duplicate_labels_removed=duplicate_labels,bbox_wh_quantiles=np.quantile(sizes,[0,.25,.5,.75,1],axis=0).tolist())
    (ROOT/'original_fixed_threshold.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
