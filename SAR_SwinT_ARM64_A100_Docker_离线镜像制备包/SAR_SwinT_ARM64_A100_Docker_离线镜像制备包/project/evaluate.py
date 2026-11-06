"""mAP50/mAP50:95 + per-class recall and false positives per image on full scenes.
No COCO dependency. Computes standard 101-point interpolated AP using global ranking;
results are HBB metrics for bounding boxes derived from quadrilaterals.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from data import parse_txt, load_gray, read_split, sample_paths
from predict import predict_full_image, restore_model


def box_iou_numpy(a,b):
    inter_x1=np.maximum(a[:,None,0],b[None,:,0]); inter_y1=np.maximum(a[:,None,1],b[None,:,1])
    inter_x2=np.minimum(a[:,None,2],b[None,:,2]); inter_y2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.maximum(inter_x2-inter_x1,0)*np.maximum(inter_y2-inter_y1,0)
    area_a=np.maximum(a[:,2]-a[:,0],0)*np.maximum(a[:,3]-a[:,1],0)
    area_b=np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)
    return inter / np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-10)


def eval_class(preds,gts,class_id,iou_thr):
    gt_map={i:np.asarray([a['box'] for a in anns if a['label']==class_id],dtype=np.float32).reshape(-1,4)
            for i,anns in enumerate(gts)}
    total=sum(len(v) for v in gt_map.values())
    if total==0: return None
    matched={i:np.zeros(len(v),bool) for i,v in gt_map.items()}
    dets=[]
    for i,pred in enumerate(preds):
        for b,s,l in zip(pred['boxes'],pred['scores'],pred['labels']):
            if int(l)==class_id: dets.append((float(s),i,np.asarray(b)))
    dets.sort(key=lambda d:-d[0]); tp=np.zeros(len(dets)); fp=np.zeros(len(dets))
    for idx,(_,i,b) in enumerate(dets):
        targets=gt_map[i]
        if len(targets)==0:fp[idx]=1;continue
        ious=box_iou_numpy(b.reshape(1,4),targets)[0]
        best=int(ious.argmax())
        if ious[best]>=iou_thr and not matched[i][best]:tp[idx]=1;matched[i][best]=True
        else:fp[idx]=1
    ctp=np.cumsum(tp); cfp=np.cumsum(fp)
    recalls=ctp/total; precs=ctp/np.maximum(ctp+cfp,1e-10)
    ap=float(np.mean([np.max(precs[recalls>=r]) if np.any(recalls>=r) else 0 for r in np.linspace(0,1,101)]))
    return {'ap':ap,'recall':float(recalls[-1]) if len(recalls) else 0,
            'tp':int(tp.sum()),'fp':int(fp.sum()),'n_gt':total}


def recall_by_area(preds,gts,iou_thr=0.5):
    """GT-based recall for area bins, matching predictions by class at the given threshold."""
    bins={'tiny_lt_16sq':[0,0], 'small_16to32sq':[0,0], 'large_ge_32sq':[0,0]}
    for pred,gt in zip(preds,gts):
        if not gt: continue
        gt_boxes=np.asarray([a['box'] for a in gt],dtype=np.float32).reshape(-1,4)
        gt_labels=np.asarray([a['label'] for a in gt],dtype=np.int64)
        matched=np.zeros(len(gt),dtype=bool)
        order=np.argsort(-pred['scores'])
        for i in order:
            if len(pred['boxes'])==0:break
            eligible=np.where((gt_labels==pred['labels'][i])&(~matched))[0]
            if not len(eligible):continue
            ious=box_iou_numpy(np.asarray(pred['boxes'][i]).reshape(1,4),gt_boxes[eligible])[0]
            ix=np.argmax(ious)
            if ious[ix]>=iou_thr:matched[eligible[ix]]=True
        areas=(gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])
        keys=np.where(areas<256,'tiny_lt_16sq',np.where(areas<1024,'small_16to32sq','large_ge_32sq'))
        for j,k in enumerate(keys):
            bins[str(k)][0]+=int(matched[j]);bins[str(k)][1]+=1
    return {k:{'matched':v[0],'gt':v[1],'recall':round(v[0]/v[1],4) if v[1] else None}
            for k,v in bins.items()}


@torch.inference_mode()
def evaluate_model(model,data_root,stems,class_names,device,tile,overlap=128,score_thresh=0.05):
    if not stems: return None
    path=Path(data_root)
    class_map={s:i+1 for i,s in enumerate(class_names)}
    preds=[];gts=[]
    for i,record in enumerate(stems):
        image_path, label_path = sample_paths(path, record)
        pred=predict_full_image(model,load_gray(image_path),device,tile,overlap,score_thresh)
        preds.append({k:v.numpy() for k,v in pred.items()})
        gts.append([{'box':a['box'],'label':class_map[a['label']]}
                    for a in parse_txt(label_path)])
        print(f"  eval {i+1}/{len(stems)} {record['image']}: {len(pred['boxes'])} detections", flush=True)
    aps50=[];aps5095=[];per_class={}
    for class_id,name in enumerate(class_names,1):
        r50=eval_class(preds,gts,class_id,.5)
        if r50 is None:continue
        avg=float(np.mean([eval_class(preds,gts,class_id,float(t))['ap'] for t in np.arange(.50,.951,.05)]))
        per_class[name]={'AP50':round(r50['ap'],4),'AP50_95':round(avg,4),
                         'recall_at_detection_threshold':round(r50['recall'],4),
                         'false_positives_per_image':round(r50['fp']/len(stems),3),
                         'gt_count':r50['n_gt']}
        aps50.append(r50['ap']);aps5095.append(avg)
    return {'mAP50':float(np.mean(aps50)) if aps50 else 0.,
            'mAP50_95':float(np.mean(aps5095)) if aps5095 else 0.,
            'num_images':len(stems),'num_evaluated_classes':len(aps50),
            'score_threshold_for_recall_and_false_positives':score_thresh,
            'per_class':per_class,'recall_by_target_area':recall_by_area(preds,gts)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--split',default='split.json')
    p.add_argument('--subset',choices=['train','val','test'],default='test')
    p.add_argument('--overlap',type=int,default=128)
    p.add_argument('--score-thresh',type=float,default=0.05)
    p.add_argument('--out',default='metrics.json')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    classes,splits=read_split(args.split)
    model,ckpt=restore_model(args.checkpoint,args.device)
    assert classes==ckpt['class_names'],'Class name order differs from checkpoint.'
    metrics=evaluate_model(model,args.data_root,splits[args.subset],classes,args.device,
                           ckpt['tile'],args.overlap,args.score_thresh)
    if metrics is None: raise SystemExit(f'Empty {args.subset} split; cannot evaluate.')
    Path(args.out).write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(metrics,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
