from pathlib import Path
import json, csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from evaluate import ROOT,REPO,read,match,write_csv

def main():
    entries=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8')); byid={e['id']:e for e in entries}
    with open(ROOT/'target_details.csv',encoding='utf-8-sig') as f: targets=list(csv.DictReader(f))
    for t in targets:
        for k in ['detected','bbox_area','mask_area','scr','local_bg_std','local_bg_gradient','width','height']:
            t[k]=float(t[k])
    unique=[t for t in targets if t['model']=='UVA_cos']
    cuts={key:np.quantile([t[key] for t in unique],[1/3,2/3]).tolist() for key in ['scr','local_bg_std','local_bg_gradient']}
    rows=[]
    for model in ['UVA_cos','UVA_linear']:
        ts=[t for t in targets if t['model']==model]
        groups={
            'bbox_area<=16':lambda t:t['bbox_area']<=16,
            'bbox_area17-64':lambda t:16<t['bbox_area']<=64,
            'bbox_area65-256':lambda t:64<t['bbox_area']<=256,
            'bbox_area>256':lambda t:t['bbox_area']>256,
        }
        for key,(a,b) in cuts.items():
            groups[key+'_low']=lambda t,key=key,a=a:t[key]<=a
            groups[key+'_mid']=lambda t,key=key,a=a,b=b:a<t[key]<=b
            groups[key+'_high']=lambda t,key=key,b=b:t[key]>b
        for group,fn in groups.items():
            selected=[t for t in ts if fn(t)]; n=len(selected); tp=sum(t['detected'] for t in selected)
            rows.append(dict(model=model,group=group,targets=n,detected=int(tp),recall=tp/n if n else None))
    write_csv(ROOT/'stratified_recall.csv',rows)
    reasons={m:{r:sum(t['reason']==r and t['model']==m for t in targets) for r in sorted({t['reason'] for t in targets})} for m in ['UVA_cos','UVA_linear']}
    (ROOT/'analysis_summary.json').write_text(json.dumps(dict(cuts=cuts,reasons=reasons,strata=rows),indent=2),encoding='utf-8')
    print(json.dumps(dict(cuts=cuts,reasons=reasons,strata=rows),indent=2))
    pred={m:json.loads((ROOT/(m+'_predictions.json')).read_text()) for m in ['UVA_cos','UVA_linear']}
    missed=[t for t in unique if not t['detected']]
    selected=[]
    selectors=[('tiny missed',sorted(missed,key=lambda t:t['bbox_area'])),
               ('low contrast missed',sorted(missed,key=lambda t:t['scr'])),
               ('clutter missed',sorted(missed,key=lambda t:-t['local_bg_std'])),
               ('localization', [t for t in missed if t['reason']=='localization_iou_0.1_to_0.5']),
               ('threshold miss',[t for t in missed if t['reason']=='confidence_below_0.25']),
               ('detected reference',sorted([t for t in unique if t['detected']],key=lambda t:-t['local_bg_std']))]
    for title,opts in selectors:
        for t in opts:
            if t['id'] not in [x[1] for x in selected]:
                selected.append((title,t['id'],int(t['gt_index']))); break
    negatives=[e for e in entries if not e['gt']]
    negatives.sort(key=lambda e:-sum(p[4]>=.25 for p in pred['UVA_cos'][e['id']]))
    for e in negatives[:4]: selected.append(('background false alarms',e['id'],None))
    out=ROOT/'cases'; out.mkdir(exist_ok=True)
    from matplotlib import font_manager
    font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),17)
    sheets=[]; caseinfo=[]
    for idx,(title,key,g) in enumerate(selected):
        e=byid[key]; arr=read(e['path']); im=Image.fromarray(arr).convert('RGB'); w,h=im.size
        panels=[]
        for m in ['UVA_cos','UVA_linear']:
            canvas=im.copy(); d=ImageDraw.Draw(canvas)
            p,pairs,_=match(e['gt'],pred[m][key]); hit={k for k,j in pairs}; matched={j for k,j in pairs}
            for k,b in enumerate(e['gt']):
                d.rectangle(b,outline=(0,255,0) if k in hit else (255,50,50),width=2)
            for j,b in enumerate(p):
                d.rectangle(tuple(b[:4]),outline=(0,200,255),width=1)
            scale=min(600/w,400/h); canvas=canvas.resize((round(w*scale),round(h*scale)))
            panel=Image.new('RGB',(620,540),'#141923'); panel.paste(canvas,((620-canvas.width)//2,38))
            d=ImageDraw.Draw(panel); d.text((10,10),f'{m}: GT {len(e["gt"])} TP {len(pairs)} FP {len(p)-len(pairs)}',font=font)
            if g is not None:
                b=e['gt'][g]; cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2; size=max(40,b[2]-b[0]+20,b[3]-b[1]+20)
                crop=im.crop((int(cx-size/2),int(cy-size/2),int(cx+size/2),int(cy+size/2))).resize((100,100),Image.Resampling.NEAREST)
                panel.paste(crop,(10,435)); feat=e['features'][g]
                d.text((120,440),f'GT #{g}: {feat["width"]}x{feat["height"]} px; SCR {feat["scr"]:.2f}',font=font)
                d.text((120,466),f'local background SD {feat["local_bg_std"]:.2f}',font=font)
            d.text((120,510),'Green=hit GT; red=missed GT; cyan=prediction',font=font)
            panels.append(panel)
        sheet=Image.new('RGB',(1240,585),'#141923'); d=ImageDraw.Draw(sheet); d.text((10,10),f'{idx+1:02d} {title}: {key}',font=font)
        for j,panel in enumerate(panels):sheet.paste(panel,(620*j,40))
        sheet.save(out/f'{idx+1:02d}_{key}.png'); sheets.append(sheet.resize((744,351)))
        caseinfo.append(dict(index=idx+1,title=title,id=key,gt_index=g,file=f'cases/{idx+1:02d}_{key}.png'))
    contact=Image.new('RGB',(1488,351*((len(sheets)+1)//2)),'white')
    for i,s in enumerate(sheets):contact.paste(s,((i%2)*744,(i//2)*351))
    contact.save(ROOT/'case_contact_sheet.jpg')
    (ROOT/'cases.json').write_text(json.dumps(caseinfo,indent=2),encoding='utf-8')

if __name__=='__main__':main()
