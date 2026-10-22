from pathlib import Path
OUT=Path(r'C:\Users\Kyson\Desktop\检测结果')
template=Path(r'C:\Users\Kyson\Desktop\新\scripts\evaluate.py').read_text(encoding='utf-8')
prepare='''def prepare(name,dest=None,limit=0):
 dest=dest if dest is not None else OUT/'results'/name
 root=Path(r'C:\\Users\\Kyson\\Desktop\\分割检测')/name
 split_file=root/'img_idx'/('test_'+name+'.txt')
 stems=[s.strip() for s in split_file.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
 assert len(stems)==len(set(stems)), 'duplicate entries in test split'
 merged=Path(r'C:\\Users\\Kyson\\Desktop\\分割检测\\yolo_dataset\\all4_segment\\images')
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
'''
start=template.index('def prepare(');end=template.index('\ndef main():',start)
code=template[:start]+prepare+template[end:]
code=code.replace("OUT=Path(r'C:\\Users\\Kyson\\Desktop\\新')","OUT=Path(r'C:\\Users\\Kyson\\Desktop\\检测结果')")
code=code.replace("choices=['ISTDD','ITSDT-15K','IRDST']","choices=['NUAA-SIRST','NUDT-SIRST']")
code=code.replace("default IRDST/ITSDT-15K=2, ISTDD=0", "default=0 for these two datasets")
code=code.replace("(2.0 if name in ['IRDST','ITSDT-15K'] else 0.0)","0.0")
hook='''   visual=res.orig_img.copy()
   if union.any():
    overlay=visual.copy();overlay[union>0]=(255,100,0);visual=cv2.addWeighted(visual,.55,overlay,.45,0)
   for x1,y1,x2,y2 in byid[rec['id']]:cv2.rectangle(visual,(int(x1),int(y1)),(int(x2-1),int(y2-1)),(0,255,0),1)
   for q in sel:
    x,y,bw,bh=q['bbox'];cv2.rectangle(visual,(int(x),int(y)),(int(x+bw),int(y+bh)),(0,0,255),1)
   cv2.putText(visual,'GT green | Pred red | Mask blue | conf 0.25',(5,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
   save(dest/'visualizations'/f"{rec['id']:04d}_{Path(rec['path']).stem}.png",visual)
'''
needle="   timings.append(res.speed);rows.append(row)"
assert needle in code
code=code.replace(needle,hook+needle)
(OUT/'scripts/run_detection.py').write_text(code,encoding='utf-8')
print('Runner saved:',OUT/'scripts/run_detection.py')
