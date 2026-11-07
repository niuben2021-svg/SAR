"""Single GPU or torchrun DDP training of Swin-T/P1-P5 Faster R-CNN on SAR TIFF/PNG labels."""
import argparse, os, random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from data import TrainTiles, collate_fn, read_split
from model import build_detector


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--split',default='split.json')
    p.add_argument('--out-dir',default='runs/swin_p1')
    p.add_argument('--swin-weights',default=None,help='Local torchvision swin_t pretrained .pth (no auto downloads)')
    p.add_argument('--tile',type=int,default=512)
    p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--batch-size',type=int,default=2,help='PER GPU')
    p.add_argument('--tiles-per-image',type=int,default=8)
    p.add_argument('--lr-backbone',type=float,default=2e-5)
    p.add_argument('--lr-head',type=float,default=1e-4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-p1',action='store_true',help='Ablation P2..P5 instead of P1..P5')
    args=p.parse_args()
    if args.tile % 32: raise SystemExit('--tile must be a multiple of 32')
    if not torch.cuda.is_available(): print('WARNING: CPU training is extremely slow.')
    is_ddp=int(os.environ.get('WORLD_SIZE','1'))>1
    rank=int(os.environ.get('RANK',0)); local_rank=int(os.environ.get('LOCAL_RANK',0))
    if is_ddp:
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
    device=torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed+rank)
    classes,splits=read_split(args.split)
    dataset=TrainTiles(args.data_root,splits['train'],classes,tile=args.tile,
                       tiles_per_image=args.tiles_per_image)
    sampler=DistributedSampler(dataset,shuffle=True) if is_ddp else None
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=sampler is None,
                      sampler=sampler,num_workers=args.workers,collate_fn=collate_fn,
                      pin_memory=torch.cuda.is_available(),drop_last=False)
    model=build_detector(len(classes)+1,image_size=args.tile,
                         swin_weights=args.swin_weights,use_p1=not args.no_p1).to(device)
    swin_params=list(model.backbone.swin.parameters())
    swin_ids={id(param) for param in swin_params}
    new_params=[p for p in model.parameters() if id(p) not in swin_ids]
    opt=torch.optim.AdamW([{'params':swin_params,'lr':args.lr_backbone},
                           {'params':new_params,'lr':args.lr_head}],weight_decay=0.05)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-6)
    use_amp=(device.type=='cuda')
    scaler=torch.cuda.amp.GradScaler(enabled=use_amp)  # compatible with PyTorch 2.3.1
    if is_ddp: model=DDP(model,device_ids=[local_rank],find_unused_parameters=True)
    out=Path(args.out_dir)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        print('classes:',classes,'train images:',len(splits['train']),'tile:',args.tile,
              'P1:',not args.no_p1,'GPUs:',int(os.environ.get('WORLD_SIZE','1')),flush=True)
        if args.swin_weights is None: print('WARNING: training Swin-T from scratch; use --swin-weights for meaningful experiments.')
    for epoch in range(args.epochs):
        if sampler is not None:sampler.set_epoch(epoch)
        model.train(); loss_total=0
        for iteration,(images,targets) in enumerate(loader):
            images=[i.to(device,non_blocking=True) for i in images]
            targets=[{k:v.to(device,non_blocking=True) for k,v in t.items()} for t in targets]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=use_amp):
                losses=model(images,targets)
                loss=sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss epoch={epoch+1} iter={iteration}: {losses}')
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
            scaler.step(opt);scaler.update()
            loss_total+=float(loss.detach())
            if rank==0 and (iteration+1)%20==0:
                print(f'epoch {epoch+1}/{args.epochs} iter {iteration+1}/{len(loader)} '
                      f'loss {float(loss):.4f}',flush=True)
        scheduler.step()
        if rank==0:
            unwrapped=model.module if is_ddp else model
            save={'model':unwrapped.state_dict(),'epoch':epoch+1,
                  'tile':args.tile,'use_p1':not args.no_p1,'class_names':classes}
            torch.save(save,out/'last.pth')
            if (epoch+1)%5==0 or epoch+1==args.epochs:
                torch.save(save,out/f'epoch_{epoch+1:03d}.pth')
            print(f'epoch {epoch+1}: train loss mean {loss_total/max(1,len(loader)):.4f}',flush=True)
        if is_ddp:dist.barrier()
    if is_ddp:dist.destroy_process_group()

if __name__=='__main__':main()
