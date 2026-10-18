#!/usr/bin/env python3
"""V20: photon-conditioned structural luminance residual (three paired ablations).

Keep legacy V15–V19 checkpoints unchanged. GT RGB/luminance is used for training
or evaluation only, never as an input to deployable reconstruction.

Actions: check, train, eval_rgb, eval_seg, all.
Requires existing train_eval_v17.py, v17_v15_compat.py, v17_v16_compat.py.
Use run_v20.ps1 on the Windows GPU machine for a one-shot experiment.
"""
import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import v17_v15_compat as v15
import v17_v16_compat as v16
import train_eval_v17 as v17

ARMS = ('edge', 'edge_ph', 'multiscale')
DEFAULT_RGB = ('raw','mle','full','adapter_ddcolor','unsharp_ddcolor',
               'edge_ddcolor','edge_ph_ddcolor','multiscale_ddcolor','ddcolor_oracle')
VOC_CLASSES = ('background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
               'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def device_of(args):
    d=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if d=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    return torch.device(d)


def write_csv(path,records):
    if not records: raise RuntimeError(f'Cannot write empty CSV {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0].keys()));wr.writeheader();wr.writerows(records)


def blur(y,sigma):
    radius=max(1,int(math.ceil(3*sigma)))
    x=torch.arange(-radius,radius+1,device=y.device,dtype=y.dtype)
    w=torch.exp(-0.5*(x/sigma).square());w=w/w.sum()
    # separable, safe for both float32 and AMP bfloat16 input
    k=(w[:,None]*w[None,:])[None,None]
    return F.conv2d(F.pad(y,(radius,)*4,mode='replicate'),k)


def gradients(y):
    return y[...,1:,:]-y[...,:-1,:],y[...,:,1:]-y[...,:,:-1]


def edge_loss(a,b):
    ay,ax=gradients(a);by,bx=gradients(b)
    return .5*((ax-bx).abs().mean()+(ay-by).abs().mean())


def highpass_error(a,b,sigma=2.):
    return F.l1_loss(a-blur(a,sigma),b-blur(b,sigma))


def photon_nll(y,k,budget,args):
    # Use FP32 even with CUDA AMP; log probabilities at B=.5 can be delicate.
    y=y.float();k=k.float()
    if args.mode=='bernoulli':
        p=(-torch.expm1(-(budget*y+args.bg)/args.cycles)).clamp(1e-7,1-1e-7)
        return (-(k*torch.log(p)+(args.cycles-k)*torch.log1p(-p))/args.cycles).mean()
    mu=(budget*y+args.bg).clamp_min(1e-7)
    return (mu-k*torch.log(mu)).mean()/max(budget,.5)


class Res(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        groups=8 if co%8==0 else 1
        self.body=nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.GroupNorm(groups,co),nn.SiLU(),
                                nn.Conv2d(co,co,3,padding=1),nn.GroupNorm(groups,co),nn.SiLU())
        self.skip=nn.Conv2d(ci,co,1)
    def forward(self,x):return self.body(x)+self.skip(x)


class StructuralResidual(nn.Module):
    """Lightweight residual U-Net, zero-init: initial prediction equals V17 Adapter.

    Inputs: five photon channels, Adapter Y, Adapter highpass, count highpass.
    Output: bounded residual in linear light. No GT enters forward().
    """
    def __init__(self,base=24,max_delta=.45):
        super().__init__();self.base=base;self.max_delta=max_delta
        self.e1=Res(8,base);self.e2=Res(base,base*2);self.mid=Res(base*2,base*4)
        self.d2=Res(base*6,base*2);self.d1=Res(base*3,base)
        self.out=nn.Conv2d(base,1,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,features,y0):
        yl=blur(y0,2.)
        raw=features[:,:1]
        x=torch.cat([features,y0,y0-yl,raw-blur(raw,2.)],dim=1)
        a=self.e1(x);b=self.e2(F.avg_pool2d(a,2));c=self.mid(F.avg_pool2d(b,2))
        d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        e=self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return (y0+self.max_delta*torch.tanh(self.out(e))).clamp(0,1)


def loss_terms(pred,ygt,y0,k,budget,args,arm):
    pix=F.l1_loss(pred,ygt)
    e=edge_loss(pred,ygt)
    hp=highpass_error(pred,ygt)
    # Discourage inventing low-frequency color/illumination through the residual.
    dc=F.l1_loss(blur(pred-y0,3.),torch.zeros_like(y0))
    total=pix+args.w_edge*e+args.w_highpass*hp+args.w_lowpass*dc
    ph=pred.new_zeros(())
    if arm in ('edge_ph','multiscale'):
        ph=photon_nll(pred,k,budget,args)
        total=total+args.w_photon*ph
    multi=pred.new_zeros(())
    if arm=='multiscale':
        # Coarse-to-fine structure constraints, independent of RGB/DeepLab.
        for s in (2,4):
            ap=F.avg_pool2d(pred,s);ag=F.avg_pool2d(ygt,s)
            multi=multi+edge_loss(ap,ag)+.5*highpass_error(ap,ag,sigma=1.)
        total=total+args.w_multiscale*multi
    return total,{'pixel':pix,'edge':e,'highpass':hp,'photon':ph,'multiscale':multi}


def ckpath(args,arm,which='best'):
    return args.out/'models'/arm/f'{which}.pt'


def config_acquisition(args):
    keys=('mode','cycles','bg','size','seed','val_count','bf_radius','bf_sigma','bf_knoise','bf_prec_cap')
    return {k:getattr(args,k) for k in keys}


def frozen_adapter(args,device):
    va=argparse.Namespace(**vars(args));va.out=args.v17_out
    return v17.load_adapter(va,device).requires_grad_(False).eval()


def load_new(args,arm,device):
    p=ckpath(args,arm)
    if not p.is_file():raise FileNotFoundError(f'Missing {p}; run V20 train first')
    ck=torch.load(p,map_location='cpu',weights_only=False)
    for k,v in config_acquisition(args).items():
        if ck['acquisition'][k]!=v:raise ValueError(f'V20 {arm} checkpoint mismatch {k}: saved {ck["acquisition"][k]} CLI {v}')
    net=StructuralResidual(ck['base'],ck['max_delta']).to(device)
    net.load_state_dict(ck['model'],strict=True)
    print(f'Loaded V20 {arm}, best epoch={ck["epoch"]} score={ck["val_score"]:.5f}',flush=True)
    return net.eval()


def load_legacy(args,device,methods):
    variants={m for m in methods if m in ('raw','mle','full')}
    result={m:v16.load_v15(args,device,m) for m in sorted(variants)}
    return result


def save_checkpoint(args,arm,net,ep,score,which,optim=None):
    p=ckpath(args,arm,which);p.parent.mkdir(parents=True,exist_ok=True)
    ck=dict(model=net.state_dict(),epoch=ep,val_score=float(score),acquisition=config_acquisition(args),
            base=args.base,max_delta=args.max_delta,arm=arm,algorithm='V20 structural luminance residual')
    if optim is not None:ck['optimizer']=optim.state_dict()
    torch.save(ck,p)


def check(args):
    dev=device_of(args)
    tr=v15.RGBImages(args.data,'train',args.size,args.seed,args.val_count)
    val=v15.RGBImages(args.data,'val',args.size,args.seed,args.val_count)
    if set(tr.paths)&set(val.paths):raise RuntimeError('Imagenette train/val overlap')
    adapter=frozen_adapter(args,dev)
    for arm in ('raw','mle','full'):
        if (args.v15_out/f'{arm}_best.pt').exists(): v16.load_v15(args,dev,arm)
    print(f'PASS: train={len(tr)}, val={len(val)}; V17 loaded, original checkpoints unchanged',flush=True)
    with torch.no_grad():
        gt,_=val[0];gt=gt[None].to(dev);v15.seed_photons(args,0,2.,salt=200000)
        features,k,_,_,_=v15.make_features(gt,2.,args)
        y=adapter(features)
        z=StructuralResidual(args.base,args.max_delta).to(dev)(features,y)
        print('Shape check:',tuple(z.shape),'finite:',bool(torch.isfinite(z).all()),flush=True)


def sample_dataset(ds,limit):
    if limit==0 or limit>=len(ds):return ds
    # Fixed, deterministic subset; no reshuffling across ablations.
    indexes=np.linspace(0,len(ds)-1,limit,dtype=int).tolist()
    return Subset(ds,indexes)


def to_tensorboard_panels(args,device,ds,adapter,nets,writer,split,ep,budgets=None,n=2):
    if writer is None or n<1:return
    with torch.no_grad():
        previous={a:net.training for a,net in nets.items()}
        for net in nets.values():net.eval()
        for idx in range(min(n,len(ds))):
            gt,_=ds[idx];gt=gt[None].to(device)
            for budget in budgets or args.tb_budgets:
                v15.seed_photons(args,idx,budget,salt=500000 if split=='train' else 200000)
                feat,k,_,_,_=v15.make_features(gt,budget,args)
                y0=adapter(feat)
                chans=[v15.linear_to_srgb(gt[0]),v15.linear_to_srgb(y0[0].repeat(3,1,1)),
                       (k[0]/args.cycles).repeat(3,1,1)]
                for arm in ARMS:
                    chans.append(v15.linear_to_srgb(nets[arm](feat,y0)[0].repeat(3,1,1)))
                panel=torch.cat(chans,dim=2).float().clamp(0,1)
                writer.add_image(f'{split}/scene_{idx:02d}_B{budget:g}_GT_Adapter_Counts_Edge_EdgePh_Multi',panel,ep)
        for a,net in nets.items():net.train(previous[a])
    writer.flush()


def validate(args,adapter,nets,ds,device):
    stats={a:[] for a in ARMS}
    for m in nets.values():m.eval()
    with torch.no_grad():
        for i in range(min(len(ds),args.val_images)):
            gt,_=ds[i];gt=gt[None].to(device);truth=v15.luminance(gt)
            for b in args.budgets:
                v15.seed_photons(args,i,b,salt=200000)
                feat,_,_,_,_=v15.make_features(gt,b,args)
                y0=adapter(feat)
                for a,net in nets.items():
                    y=net(feat,y0)
                    l1=F.l1_loss(y,truth).item();ed=edge_loss(y,truth).item();mse=F.mse_loss(y,truth).item()
                    hp=highpass_error(y,truth).item()
                    # fixed scalar checkpoint-selection criterion, independent of VOC oracles
                    stats[a].append((b,l1,ed,mse,hp))
    return {a:dict(val_l1=float(np.mean([x[1] for x in ls])),
                   val_edge=float(np.mean([x[2] for x in ls])),
                   val_mse=float(np.mean([x[3] for x in ls])),
                   val_highpass=float(np.mean([x[4] for x in ls]))) for a,ls in stats.items()}


def train(args):
    seed_all(args.seed);device=device_of(args)
    check(args)
    ds=v15.RGBImages(args.data,'train',args.size,args.seed,args.val_count)
    dv=v15.RGBImages(args.data,'val',args.size,args.seed,args.val_count)
    ds=sample_dataset(ds,args.train_images)
    adapter=frozen_adapter(args,device)
    initial=StructuralResidual(args.base,args.max_delta).state_dict()
    nets={};opts={};scalers={}
    for arm in ARMS:
        net=StructuralResidual(args.base,args.max_delta).to(device)
        net.load_state_dict(initial)
        nets[arm]=net;opts[arm]=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
        scalers[arm]=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and args.amp=='fp16')
    enabled=args.amp!='none' and device.type=='cuda'
    amp_dtype=torch.float16 if args.amp=='fp16' else torch.bfloat16
    writer=None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(args.out/'tb'))
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'train_config.json').write_text(json.dumps({k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},indent=2),encoding='utf8')
    history=[];best={a:float('inf') for a in ARMS};stale=0;glob_step=0
    print(f'V20: {len(ds)} train / {len(dv)} val, 3 paired arms, {args.epochs} epochs max',flush=True)
    try:
        to_tensorboard_panels(args,device,ds,adapter,nets,writer,'train',0,n=args.tb_train_images)
        to_tensorboard_panels(args,device,dv,adapter,nets,writer,'val',0,n=args.tb_val_images)
        for epoch in range(args.epochs):
            start=time.perf_counter()
            if hasattr(ds,'dataset'):ds.dataset.epoch=epoch
            else:ds.epoch=epoch
            loader=DataLoader(ds,batch_size=args.batch,shuffle=True,num_workers=args.workers,
                              pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(args.seed+epoch))
            acc={a:0. for a in ARMS};seen=0
            for bi,(gt,_) in enumerate(loader):
                if args.max_batches and bi>=args.max_batches:break
                gt=gt.to(device,non_blocking=True);n=gt.shape[0]
                budget=args.budgets[(epoch*101+bi)%len(args.budgets)]
                v15.seed_photons(args,bi+epoch*100000,budget)
                with torch.no_grad():
                    feats,k,_,_,_=v15.make_features(gt,budget,args)
                    y0=adapter(feats)
                    ygt=v15.luminance(gt)
                # Same photon realization and same initial network weights for ALL ablations.
                for arm in ARMS:
                    net=nets[arm];net.train()
                    with torch.amp.autocast(device_type=device.type,enabled=enabled,dtype=amp_dtype):
                        yp=net(feats,y0)
                        loss,parts=loss_terms(yp,ygt,y0,k,budget,args,arm)
                    if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {arm} loss at step {glob_step}, budget {budget}')
                    opt=opts[arm];scaler=scalers[arm]
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(net.parameters(),1.)
                    scaler.step(opt);scaler.update()
                    acc[arm]+=loss.detach().item()*n
                    if writer and (glob_step%args.log_every==0):
                        writer.add_scalar(f'train_batch/{arm}_total',loss.detach().item(),glob_step)
                        writer.add_scalar(f'train_batch/{arm}_edge',parts['edge'].detach().item(),glob_step)
                seen+=n;glob_step+=1
                if bi%args.log_every==0:
                    print(f'epoch={epoch+1}/{args.epochs} batch={bi+1}/{len(loader)} '
                          f'edge={acc["edge"]/seen:.4f} edge_ph={acc["edge_ph"]/seen:.4f} multi={acc["multiscale"]/seen:.4f}',flush=True)
                if writer and args.tb_train_every and glob_step%args.tb_train_every==0:
                    to_tensorboard_panels(args,device,ds,adapter,nets,writer,'train',glob_step,n=args.tb_train_images)
            if seen==0:raise RuntimeError('No training batches processed')
            metrics=validate(args,adapter,nets,dv,device)
            changed=False
            for a in ARMS:
                s=metrics[a];score=s['val_l1']+args.select_edge*s['val_edge']
                s['val_score']=score
                row=dict(epoch=epoch+1,arm=a,train_loss=acc[a]/seen,seconds=time.perf_counter()-start,**s)
                history.append(row)
                save_checkpoint(args,a,nets[a],epoch+1,score,'last',opts[a])
                if score<best[a]-args.min_improve:
                    best[a]=score;changed=True
                    save_checkpoint(args,a,nets[a],epoch+1,score,'best')
                if writer:
                    for key,v in row.items():
                        if key not in ('arm','epoch'):
                            writer.add_scalar(f'epoch/{a}_{key}',float(v),epoch+1)
            write_csv(args.out/'train_history.csv',history)
            if writer:
                to_tensorboard_panels(args,device,dv,adapter,nets,writer,'val',epoch+1,n=args.tb_val_images)
                writer.flush()
            print(f'epoch {epoch+1} completed ({time.perf_counter()-start:.1f}s): '+
                  ', '.join(f'{a}:MAE={metrics[a]["val_l1"]:.4f} edge={metrics[a]["val_edge"]:.4f} best={best[a]:.4f}' for a in ARMS),flush=True)
            stale=0 if changed else stale+1
            if stale>=args.patience:
                print(f'EARLY STOP: none of 3 arms improved in {args.patience} epochs',flush=True)
                break
    finally:
        if writer:writer.close()
    print('V20 training finished. Best checkpoints:',*(str(ckpath(args,a)) for a in ARMS),flush=True)


@torch.no_grad()
def predictions(args,gt,idx,budget,adapter,nets,legacy,colorizer,methods):
    # The exact same observed count image is reused across methods.
    v15.seed_photons(args,idx,budget,salt=300000)
    feat,k,est,var,bf=v15.make_features(gt,budget,args)
    ya=adapter(feat).clamp(0,1)
    out={};ys={'adapter':ya}
    for arm in ARMS:
        if arm+'_ddcolor' in methods or arm in methods:
            ys[arm]=nets[arm](feat,ya)
    for m,model in legacy.items():out[m]=model(v16.make_variant(feat,m))
    for m,y in ys.items():
        if m in ARMS and m in methods:out[m]=y.repeat(1,3,1,1)
        if f'{m}_ddcolor' in methods:
            if colorizer is None:raise RuntimeError('DDColor missing')
            out[f'{m}_ddcolor']=v16.ddcolorize(colorizer,y,gt.device)
    if 'unsharp_ddcolor' in methods:
        sharpened=(ya+args.sharpen*(ya-blur(ya,2.))).clamp(0,1)
        out['unsharp_ddcolor']=v16.ddcolorize(colorizer,sharpened,gt.device)
    if 'ddcolor_oracle' in methods:
        out['ddcolor_oracle']=v16.ddcolorize(colorizer,v15.luminance(gt),gt.device)
    return out,dict(k=k,est=est,bf=bf,adapter=ya,targets=ys)


def build_inference(args,device,methods):
    adapter=frozen_adapter(args,device)
    active=[a for a in ARMS if a in methods or a+'_ddcolor' in methods]
    nets={a:load_new(args,a,device) for a in active}
    legacy=load_legacy(args,device,methods)
    needs_ddcolor=bool(set(methods)&{'adapter_ddcolor','unsharp_ddcolor','edge_ddcolor',
                   'edge_ph_ddcolor','multiscale_ddcolor','ddcolor_oracle'})
    colorizer=v16.load_ddcolor(args,device)[0] if needs_ddcolor else None
    return adapter,nets,legacy,colorizer


def edge_metrics(y,gt):
    a=y.float();b=v15.luminance(gt).float()
    return dict(luma_mae=float(F.l1_loss(a,b)),luma_mse=float(F.mse_loss(a,b)),
                luma_edge_mae=float(edge_loss(a,b)),luma_highpass_mae=float(highpass_error(a,b)))


def rgb_eval(args):
    device=device_of(args)
    ds=v15.RGBImages(args.data,args.split,args.size,args.seed,args.val_count)
    methods=list(args.methods)
    adapter,nets,legacy,colorizer=build_inference(args,device,methods)
    n=min(len(ds),args.eval_images) if args.eval_images else len(ds)
    out_dir=args.out/f'rgb_{args.split}'
    out_dir.mkdir(parents=True,exist_ok=True)
    records=[];luma=[]
    for idx in range(n):
        gt,source=ds[idx];gt=gt[None].to(device)
        panels=[]
        for budget in args.budgets:
            outs,inter=predictions(args,gt,idx,budget,adapter,nets,legacy,colorizer,methods)
            for name,rgb in outs.items():
                if name in ARMS:continue
                mm=v16.compute_metrics(rgb,gt)
                mm['measurement_nll']=float(photon_nll(v15.luminance(rgb),inter['k'],budget,args))
                records.append(dict(image_index=idx,scene=source,budget_total=budget,method=name,
                                    oracle=int(name=='ddcolor_oracle'),**mm))
            for name,y in inter['targets'].items():
                luma.append(dict(image_index=idx,scene=source,budget_total=budget,method=name,
                                 **edge_metrics(y,gt)))
            if idx<args.gallery_n:
                cols=[('Counts',v16.to_pil_srgb((inter['k']/3).clamp(0,1))),
                      ('Adapter Y',v16.to_pil_srgb(inter['adapter'])),
                      ('RGB GT',v16.to_pil_srgb(gt))]
                cols.extend((m,v16.to_pil_srgb(outs[m])) for m in methods if m in outs)
                panels.append((budget,cols))
        if panels:v16.save_gallery(panels,args.size,out_dir/f'{idx:04d}.png')
        if (idx+1)%10==0 or idx+1==n:print('RGB',idx+1,'/',n,flush=True)
    write_csv(out_dir/'rgb_per_image.csv',records)
    write_csv(out_dir/'luma_per_image.csv',luma)
    summaries=[]
    for budget in args.budgets:
        for method in [m for m in methods if m not in ARMS]:
            v=[r for r in records if r['budget_total']==budget and r['method']==method]
            if not v:continue
            summaries.append(dict(budget_total=budget,method=method,images=len(v),oracle=int(method=='ddcolor_oracle'),
                                  **{key:float(np.mean([r[key] for r in v])) for key in
                                    ('psnr_srgb','ssim_srgb','lab_ab_error','delta_e76','linear_luma_mse','measurement_nll')}))
    write_csv(out_dir/'rgb_summary.csv',summaries)
    ls=[]
    for b in args.budgets:
        for name in ('adapter',)+ARMS:
            rows=[r for r in luma if r['budget_total']==b and r['method']==name]
            if rows:
                ls.append(dict(budget_total=b,method=name,images=len(rows),**{
                    key:float(np.mean([r[key] for r in rows]))
                    for key in ('luma_mae','luma_mse','luma_edge_mae','luma_highpass_mae')}))
    write_csv(out_dir/'luma_summary.csv',ls)
    print('Saved RGB benchmark:',out_dir,flush=True)


def voc_color_table():
    tab=np.zeros((256,3),np.uint8)
    for l in range(21):
        v=l
        for bit in range(8):
            for c in range(3):tab[l,c]|=((v>>c)&1)<<(7-bit)
            v>>=3
    tab[255]=[255]*3
    return tab

VOC_PALETTE=voc_color_table()


def mask_pil(mask):return Image.fromarray(VOC_PALETTE[np.asarray(mask,dtype=np.uint8)],'RGB')


def crop_voc(rgb,mask,side):
    w,h=rgb.size;s=min(w,h);left=(w-s)//2;top=(h-s)//2
    rect=(left,top,left+s,top+s)
    return rgb.convert('RGB').crop(rect).resize((side,side),Image.Resampling.BICUBIC),\
           mask.crop(rect).resize((side,side),Image.Resampling.NEAREST)


def pil_tensor(im,device):
    arr=np.asarray(im,dtype=np.float32).copy()
    return torch.from_numpy(arr).permute(2,0,1)[None].to(device)/255.


def load_segmentor(name,device):
    from torchvision.models.segmentation import (deeplabv3_resnet50,DeepLabV3_ResNet50_Weights,
            deeplabv3_mobilenet_v3_large,DeepLabV3_MobileNet_V3_Large_Weights)
    if name=='deeplab_resnet50':
        weights=DeepLabV3_ResNet50_Weights.DEFAULT
        model=deeplabv3_resnet50(weights=weights)
    else:
        weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model=deeplabv3_mobilenet_v3_large(weights=weights)
    return model.to(device).eval(),weights.transforms()


@torch.no_grad()
def segment(model,transform,srgb,shape):
    x=transform(srgb[0].float().clamp(0,1)).unsqueeze(0)
    logits=model(x)['out'].float()
    return F.interpolate(logits,size=shape,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)


def update_confusion(cm,pred,truth):
    valid=(truth!=255)&(truth>=0)&(truth<21)&(pred>=0)&(pred<21)
    ids=truth[valid].astype(np.int64)*21+pred[valid].astype(np.int64)
    cm+=np.bincount(ids,minlength=21*21).reshape(21,21)


def segmentation_scores(cm,gt_present):
    tp=np.diag(cm).astype(float)
    union=cm.sum(0)+cm.sum(1)-tp
    keep=gt_present & (union>0)
    fg=keep.copy();fg[0]=False
    vals=tp/np.maximum(union,1)
    return dict(miou_gt_present=float(vals[keep].mean()) if keep.any() else float('nan'),
                foreground_miou=float(vals[fg].mean()) if fg.any() else float('nan'),
                pixel_accuracy=float(tp.sum()/max(cm.sum(),1)),present_classes=int(keep.sum()))


def seg_gallery(path,rows,methods,tile=160,budget=2):
    if not rows:return
    cols=['VOC GT','DeepLab GT']+methods
    pad=10;header=58;rh=22;gap=8
    width=pad*2+len(cols)*(tile+gap)
    height=header+len(rows)*(rh+2*tile+gap*2+rh)+pad
    canvas=Image.new('RGB',(width,height),'white');drawer=ImageDraw.Draw(canvas)
    drawer.text((pad,5),f'VOC segmentation | B={budget:g} | RGB top / segmentation bottom | official DeepLab',fill='black')
    for j,col in enumerate(cols):drawer.text((pad+j*(tile+gap),33),col[:20],fill='black')
    y=header
    for idx,rgb_images,masks in rows:
        drawer.text((pad,y),f'VOC index {idx}',fill='black');y+=rh
        for j,(im,ma) in enumerate(zip(rgb_images,masks)):
            x=pad+j*(tile+gap)
            canvas.paste(im.resize((tile,tile),Image.Resampling.BICUBIC),(x,y))
            canvas.paste(mask_pil(ma).resize((tile,tile),Image.Resampling.NEAREST),(x,y+tile+gap))
        y+=2*tile+gap*2+rh
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def seg_eval(args):
    from torchvision.datasets import VOCSegmentation
    dev=device_of(args)
    ds=VOCSegmentation(root=str(args.voc_root),year='2012',image_set='val',download=False)
    model,transform=load_segmentor(args.segmentor,dev)
    methods=list(args.seg_methods) if args.seg_methods else list(args.methods)
    adapter,nets,legacy,colorizer=build_inference(args,dev,methods)
    start=args.skip_voc
    n=(len(ds)-start) if args.seg_eval_images==0 else min(len(ds)-start,args.seg_eval_images)
    if n<=0:raise ValueError('No VOC validation images after skip')
    save=args.out/'seg_voc';save.mkdir(parents=True,exist_ok=True)
    seg_budgets=args.seg_budgets if args.seg_budgets else args.budgets
    cms={(m,b):np.zeros((21,21),np.int64) for m in ['GT_RGB']+methods for b in seg_budgets}
    pairs=[];galleries={b:[] for b in seg_budgets}
    for j,idx in enumerate(range(start,start+n)):
        original,mask=ds[idx]
        crop,mask=crop_voc(original,mask,args.size)
        truth=np.asarray(mask,dtype=np.uint8)
        srgb=pil_tensor(crop,dev)
        linear=v15.srgb_to_linear(srgb)
        gt_mask=segment(model,transform,srgb,truth.shape)
        for budget in (args.seg_budgets if args.seg_budgets else args.budgets):
            outs,info=predictions(args,linear,idx,budget,adapter,nets,legacy,colorizer,methods)
            pred={'GT_RGB':gt_mask}
            displays={'GT_RGB':crop}
            for m in methods:
                rgb=outs[m]
                srgb_m=v15.linear_to_srgb(rgb)
                pred[m]=segment(model,transform,srgb_m,truth.shape)
                displays[m]=v16.to_pil_srgb(rgb)
            for m,ma in pred.items():
                cm=np.zeros((21,21),np.int64);update_confusion(cm,ma,truth)
                update_confusion(cms[(m,budget)],ma,truth)
                present=np.bincount(truth[truth<21],minlength=21)>0
                pairs.append(dict(voc_index=idx,budget_total=budget,method=m,**segmentation_scores(cm,present)))
            if len(galleries[budget])<args.gallery_n:
                rgbrow=[crop,crop]+[displays[m] for m in methods]
                maskrow=[truth,gt_mask]+[pred[m] for m in methods]
                galleries[budget].append((idx,rgbrow,maskrow))
        if (j+1)%10==0 or j+1==n:print('VOC',j+1,'/',n,flush=True)
    gt_present=cms[('GT_RGB',seg_budgets[0])].sum(1)>0
    summ=[]
    for b in seg_budgets:
        for m in ['GT_RGB']+methods:
            summ.append(dict(budget_total=b,method=m,images=n,oracle=int(m=='ddcolor_oracle'),
                             **segmentation_scores(cms[(m,b)],gt_present)))
        seg_gallery(save/f'seg_gallery_B{b:g}.png',galleries[b],methods,args.seg_tile,b)
    write_csv(save/'seg_summary.csv',summ)
    write_csv(save/'seg_per_image.csv',pairs)
    np.savez_compressed(save/'seg_confusions.npz',**{f'{m}_B{b:g}':v for (m,b),v in cms.items()})
    print('Saved official VOC benchmark:',save,flush=True)



def report(args):
    """A numerical summary, not a claim of statistical significance."""
    rp=args.out/f'rgb_{args.split}'/'rgb_summary.csv'
    lp=args.out/f'rgb_{args.split}'/'luma_summary.csv'
    sp=args.out/'seg_voc'/'seg_summary.csv'
    for path in (rp,lp,sp):
        if not path.is_file():raise FileNotFoundError(f'Missing benchmark results: {path}')
    def records(p):
        with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
    rs=records(rp);ls=records(lp);ss=records(sp)
    def get(rows,name,b,key):
        r=[x for x in rows if x['method']==name and abs(float(x['budget_total'])-b)<1e-6]
        return float(r[0][key]) if r else float('nan')
    lines=['V20 UNIFIED REPORT','Validation-selected best V20 models; GT RGB used only for supervision/evaluation.',
           'Numbers are exploratory if --eval-images/--seg-eval-images are small.',
           'DDColor Oracle uses GT luminance; NOT a deployable baseline.','']
    lines+=['RGB TEST (PSNR dB, delta vs V15 Full)']
    for b in args.budgets:
        baseline=get(rs,'full',b,'psnr_srgb')
        for arm in ARMS:
            meth=f'{arm}_ddcolor';value=get(rs,meth,b,'psnr_srgb')
            lines.append(f'B={b:g} {meth}: PSNR={value:.3f} delta_full={value-baseline:+.3f}')
    lines+=['','LUMINANCE TEST (high-frequency L1 difference vs V17 Adapter)']
    for b in args.budgets:
        ref=get(ls,'adapter',b,'luma_highpass_mae')
        for a in ARMS:
            x=get(ls,a,b,'luma_highpass_mae')
            lines.append(f'B={b:g} {a}: highpass_mae={x:.5f} delta_adapter={x-ref:+.5f}')
    lines+=['','VOC SEGMENTATION (mIoU, delta vs Adapter-DDColor)']
    for b in args.seg_budgets:
        base=get(ss,'adapter_ddcolor',b,'miou_gt_present')
        for a in ARMS:
            value=get(ss,f'{a}_ddcolor',b,'miou_gt_present')
            lines.append(f'B={b:g} {a}: mIoU={value:.4f} delta_adapter={value-base:+.4f}')
    lines+=['','Interpretation: a useful V20 must improve held-out luminance high-frequency fidelity',
            'AND RGB or segmentation without introducing measurement-inconsistent artifacts.',
            'Small-sample gaps are NOT significant by themselves; confirm on full test set.']
    output=args.out/'V20_DECISION_REPORT.txt';output.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines[-12:]))
    print('Saved one-page decision report:',output,flush=True)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','train','eval_rgb','eval_seg','report','all'])
    p.add_argument('--data',type=Path,default=Path('data/imagenette2-160'))
    p.add_argument('--voc-root',type=Path,default=Path('data/VOC'))
    p.add_argument('--v15-out',type=Path,default=Path('runs/v15_64'))
    p.add_argument('--v17-out',type=Path,default=Path('runs/v17_adapter'))
    p.add_argument('--out',type=Path,default=Path('runs/v20_structural'))
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--mode',choices=['bernoulli','poisson'],default='bernoulli')
    p.add_argument('--cycles',type=int,default=64);p.add_argument('--bg',type=float,default=.01)
    p.add_argument('--budgets',type=float,nargs='+',default=[.5,1.,2.,4.,8.])
    p.add_argument('--size',type=int,default=128);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--val-count',type=int,default=500)
    p.add_argument('--bf-radius',type=int,default=3);p.add_argument('--bf-sigma',type=float,default=1.5)
    p.add_argument('--bf-knoise',type=float,default=2.);p.add_argument('--bf-prec-cap',type=float,default=4.)
    p.add_argument('--base',type=int,default=24);p.add_argument('--max-delta',type=float,default=.45)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--train-images',type=int,default=0,help='0=all; fixed subset for rapid trial')
    p.add_argument('--max-batches',type=int,default=0,help='0=full epoch; nonzero for smoke')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--lr',type=float,default=0.0002)
    p.add_argument('--amp',choices=['none','fp16','bf16'],default='fp16')
    p.add_argument('--val-images',type=int,default=24);p.add_argument('--patience',type=int,default=2)
    p.add_argument('--min-improve',type=float,default=.00002)
    p.add_argument('--w-edge',type=float,default=.65);p.add_argument('--w-highpass',type=float,default=.4)
    p.add_argument('--w-lowpass',type=float,default=.08);p.add_argument('--w-photon',type=float,default=.03)
    p.add_argument('--w-multiscale',type=float,default=.15);p.add_argument('--select-edge',type=float,default=.7)
    p.add_argument('--tensorboard',action='store_true');p.add_argument('--tb-train-images',type=int,default=2)
    p.add_argument('--tb-val-images',type=int,default=3);p.add_argument('--tb-train-every',type=int,default=300)
    p.add_argument('--tb-budgets',type=float,nargs='+',default=[.5,2.,8.]);p.add_argument('--log-every',type=int,default=40)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--eval-images',type=int,default=40,help='0 = full held-out test / VOC val after skip')
    p.add_argument('--gallery-n',type=int,default=6);p.add_argument('--seg-tile',type=int,default=160)
    p.add_argument('--seg-eval-images',type=int,default=20,help='VOC samples after skip; 0=full')
    p.add_argument('--seg-budgets',type=float,nargs='+',default=[.5,2.,8.])
    p.add_argument('--seg-methods',nargs='+',choices=DEFAULT_RGB,default=['full','adapter_ddcolor','edge_ddcolor','edge_ph_ddcolor','multiscale_ddcolor','ddcolor_oracle'])
    p.add_argument('--skip-voc',type=int,default=6,help='Exclude first 6 VOC val images used in earlier V19 checkpoint selection')
    p.add_argument('--segmentor',choices=['deeplab_resnet50','deeplab_mobilenet'],default='deeplab_resnet50')
    p.add_argument('--sharpen',type=float,default=.5)
    p.add_argument('--ddcolor-repo',type=Path,default=Path('external/DDColor'))
    p.add_argument('--ddcolor-model',default='piddnad/ddcolor_paper_tiny')
    p.add_argument('--ddcolor-weights',type=Path,default=None)
    p.add_argument('--ddcolor-model-size',choices=['tiny','large'],default='tiny')
    p.add_argument('--ddcolor-size',type=int,default=256)
    p.add_argument('--methods',nargs='+',default=list(DEFAULT_RGB),choices=DEFAULT_RGB)
    return p


def main():
    p=parser();args=p.parse_args()
    if args.size%4 or args.size<=8:p.error('--size must be a multiple of four and > 8')
    if args.base%8:p.error('--base must be a multiple of 8')
    if args.epochs<1 or args.batch<1 or args.val_images<1 or args.patience<1:p.error('positive epochs,batch,val-images,patience required')
    if args.eval_images<0 or args.seg_eval_images<0 or args.skip_voc<0 or args.gallery_n<0 or args.train_images<0:p.error('negative image count')
    if any(b<=0 for b in args.budgets + args.seg_budgets):p.error('budgets must be positive')
    if args.action=='check':check(args)
    elif args.action=='train':train(args)
    elif args.action=='eval_rgb':rgb_eval(args)
    elif args.action=='eval_seg':seg_eval(args)
    elif args.action=='report':report(args)
    else:
        train(args);rgb_eval(args);seg_eval(args);report(args)

if __name__=='__main__':main()
