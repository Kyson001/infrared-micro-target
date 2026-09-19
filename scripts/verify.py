import json
import numpy as np
from evaluate import *

def main():
    entries=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    diagnostics=[]; bgrows=[]
    for name in MODELS:
        preds=json.loads((ROOT/(name+'_predictions.json')).read_text())
        for subset,es in [('v2_full',entries),('v1',[e for e in entries if e['v1']]),('new_targets',[e for e in entries if not e['v1'] and e['gt']])]:
            for iou in [.1,.25,.5]:
                tp=fp=fn=0
                for e in es:
                    p,pairs,_=match(e['gt'],preds[e['id']],.25,iou); tp+=len(pairs); fp+=len(p)-len(pairs); fn+=len(e['gt'])-len(pairs)
                diagnostics.append(dict(model=name,subset=subset,criterion=f'IoU>={iou}',tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp),recall=tp/(tp+fn)))
            tp=fp=fn=0; ratios=[]
            for e in es:
                gt=np.array(e['gt']).reshape(-1,4); p=np.array([p for p in preds[e['id']] if p[4]>=.25]).reshape(-1,5)
                p=p[np.argsort(-p[:,4])]; gc=(gt[:,:2]+gt[:,2:])/2; pc=(p[:,:2]+p[:,2:4])/2
                dist=np.linalg.norm(gc[:,None,:]-pc[None,:,:],axis=2); used=set()
                for j in range(len(p)):
                    options=[k for k in range(len(gt)) if k not in used and dist[k,j]<=3]
                    if options:
                        k=min(options,key=lambda k:dist[k,j]); used.add(k)
                        ratios.append(float(np.prod(p[j,2:4]-p[j,:2])/np.prod(gt[k,2:]-gt[k,:2])))
                tp+=len(used); fp+=len(p)-len(used); fn+=len(gt)-len(used)
            diagnostics.append(dict(model=name,subset=subset,criterion='center_distance<=3px',tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp),recall=tp/(tp+fn)))
            print(name,subset,'center-matched median predicted/GT area',np.median(ratios) if ratios else None,flush=True)
        for e in entries:
            if e['gt']:continue
            im=read(e['path']).astype(float); grad=np.hypot(cv2.Sobel(im,cv2.CV_64F,1,0),cv2.Sobel(im,cv2.CV_64F,0,1))
            bgrows.append(dict(model=name,id=e['id'],std=float(im.std()),gradient=float(grad.mean()),fp=sum(p[4]>=.25 for p in preds[e['id']])))
    write_csv(ROOT/'localization_diagnostics.csv',diagnostics); write_csv(ROOT/'background_details.csv',bgrows)
    print('DIAGNOSTICS',json.dumps(diagnostics),flush=True)
    if ORIGINAL_DATA is None:
        print('Skipped original validation: set UVA_ORIGINAL_DATA to its dataset directory.')
        return
    results={}
    for name,path in MODELS.items():
        model=YOLO(str(path))
        r=model.val(data=str(ORIGINAL_DATA/'dataset.yaml'),imgsz=1024,batch=8,device=DEVICE,workers=0,
                    plots=False,project=str(ROOT/'original_validation'),name=name,exist_ok=True,verbose=False)
        results[name]={k:float(v) for k,v in r.results_dict.items()}
        (ROOT/'original_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':main()
