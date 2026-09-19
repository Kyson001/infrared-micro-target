"""Run either uploaded best model on an image, directory, or video."""
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

REPO=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['UVA','UVA_no_cosine'],default='UVA')
    p.add_argument('--source',default=str(REPO/'datasets/sirst-v2/mixed/Misc_1.png'))
    p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--device',default='0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,default=REPO/'results/inference')
    args=p.parse_args()
    model=YOLO(str(REPO/'models'/args.model/'best.pt'))
    count=0
    for r in model.predict(source=args.source,imgsz=1024,batch=8,rect=False,conf=args.conf,
                           iou=.7,max_det=300,device=args.device,half=False,stream=True,
                           save=True,save_txt=True,save_conf=True,project=str(args.output),name=args.model):
        count+=1
        print(f'{Path(r.path).name}: {len(r.boxes)} predictions')
    print(f'Finished {count} images/frames; output root: {args.output}')

if __name__=='__main__':main()
