"""Whole-image inference by overlapped native-resolution tiles and global per-class NMS."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import batched_nms
from model import build_detector
from data import load_gray, to_tensor


def tile_starts(length, size, step):
    if length < size: raise ValueError(f'image side {length} shorter than tile {size}')
    values=list(range(0,max(1,length-size+1),step))
    if values[-1] != length-size: values.append(length-size)
    return values


@torch.inference_mode()
def predict_full_image(model, gray, device, tile=512, overlap=128,
                       score_thresh=0.25, nms_iou=0.5):
    h,w=gray.shape
    if overlap<0 or overlap>=tile: raise ValueError('Overlap must satisfy 0 <= overlap < tile')
    all_boxes=[]; all_scores=[]; all_labels=[]
    for y0 in tile_starts(h,tile,tile-overlap):
        for x0 in tile_starts(w,tile,tile-overlap):
            crop=gray[y0:y0+tile,x0:x0+tile]
            inp=to_tensor(crop).to(device)
            out=model([inp])[0]
            keep=out['scores'] >= score_thresh
            if keep.sum() == 0: continue
            boxes=out['boxes'][keep].cpu(); scores=out['scores'][keep].cpu(); labels=out['labels'][keep].cpu()
            boxes[:,[0,2]] += x0; boxes[:,[1,3]] += y0
            all_boxes.append(boxes); all_scores.append(scores); all_labels.append(labels)
    if not all_boxes:
        return {'boxes':torch.empty((0,4)), 'scores':torch.empty((0,)), 'labels':torch.empty((0,),dtype=torch.long)}
    boxes=torch.cat(all_boxes); scores=torch.cat(all_scores); labels=torch.cat(all_labels)
    keep=batched_nms(boxes,scores,labels,nms_iou)[:500]
    return {'boxes':boxes[keep], 'scores':scores[keep], 'labels':labels[keep]}


def restore_model(ckpt_path,device):
    checkpoint=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    model=build_detector(num_classes=len(checkpoint['class_names'])+1,
                         image_size=checkpoint['tile'],use_p1=checkpoint['use_p1'])
    model.load_state_dict(checkpoint['model'],strict=True)
    model.to(device).eval()
    return model,checkpoint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--out',default='prediction.json')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--overlap',type=int,default=128)
    p.add_argument('--score-thresh',type=float,default=0.25)
    args=p.parse_args()
    model,ckpt=restore_model(args.checkpoint,args.device)
    pred=predict_full_image(model,load_gray(args.image),args.device,ckpt['tile'],args.overlap,args.score_thresh)
    names=ckpt['class_names']
    detections=[{'box':list(map(float,b.tolist())), 'score':float(s), 'class':names[int(l)-1]}
                for b,s,l in zip(pred['boxes'],pred['scores'],pred['labels'])]
    Path(args.out).write_text(json.dumps(detections,ensure_ascii=False,indent=2),encoding='utf8')
    print('Detections:',len(detections),'Saved:',args.out)

if __name__=='__main__':main()
