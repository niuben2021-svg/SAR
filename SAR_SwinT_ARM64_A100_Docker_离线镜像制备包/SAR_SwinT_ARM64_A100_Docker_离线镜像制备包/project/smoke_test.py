"""One real user-supplied annotation -> train/eval forward; not a precision evaluation."""
import argparse
from pathlib import Path
import torch
from data import load_gray, parse_txt, annotations_in_crop, to_tensor, find_partition_samples, sample_paths
from model import build_detector


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--tile',type=int,default=128)
    p.add_argument('--device',default='cpu')
    p.add_argument('--no-p1',action='store_true')
    p.add_argument('--backward',action='store_true',help='also do one optimizer step')
    args=p.parse_args()
    torch.set_num_threads(2)
    root=Path(args.data_root)
    samples=find_partition_samples(root, 'trainval')
    path=txt=None
    for record in samples:
        candidate_image, candidate_txt = sample_paths(root, record)
        if parse_txt(candidate_txt):
            path,txt=candidate_image,candidate_txt
            break
    if path is None:raise SystemExit('No annotated training example in trainval')
    gray=load_gray(path); anns=parse_txt(txt)
    labels=sorted({a['label'] for a in anns})
    box=anns[0]['box']; h,w=gray.shape
    s=args.tile
    x0=int(max(0,min(w-s,(box[0]+box[2])/2-s/2)))
    y0=int(max(0,min(h-s,(box[1]+box[3])/2-s/2)))
    image=to_tensor(gray[y0:y0+s,x0:x0+s]).to(args.device)
    target=annotations_in_crop(anns,{l:i+1 for i,l in enumerate(labels)},x0,y0,s)
    target={k:v.to(args.device) for k,v in target.items()}
    print('Input image:',path.name,'source size:',(w,h),'crop:',(x0,y0,s,s),flush=True)
    print('Labels:',labels,'crop boxes:',target['boxes'].tolist(),flush=True)
    model=build_detector(len(labels)+1,image_size=s,use_p1=not args.no_p1).to(args.device)
    model.train()
    losses=model([image],[target])
    print('TRAIN FORWARD LOSSES:',{k:round(float(v.detach()),5) for k,v in losses.items()},flush=True)
    if args.backward:
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        optimizer.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        optimizer.step()
        print('BACKWARD + OPTIMIZER STEP: OK',flush=True)
    model.eval()
    with torch.inference_mode():
        pred=model([image])[0]
    print('INFERENCE FORWARD OK. Number of untrained detections:',len(pred['boxes']),flush=True)
    print('This is a random initialized model. No valid accuracy can be inferred.',flush=True)

if __name__=='__main__':main()
