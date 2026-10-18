#!/usr/bin/env python3
"""V19: controlled VOC fine-tuning of a V18 photon-conditioned RGB latent bridge.

A: VOC reconstruction-only; B: same VOC images + frozen DeepLab semantic loss.
All trainable weights: bridge only. Segmentation backbone, RGB VAE, V17 adapter frozen.
VOC train is NEVER used for validation. Evaluation always reuses the SAME photons.

Example: python train_eval_v19.py train --arm B --out runs/v19_seg --v18-out runs/v18_bridge_fast ...
"""
import argparse, contextlib, csv, json, math, random, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import train_eval_v18 as v18
import eval_v18_segmentation as segutil

V19_NAMES = ('v19_a', 'v19_b', 'v19_a_fused', 'v19_b_fused')
DEFAULT_METHODS = ('raw', 'mle', 'full', 'bridge', 'bridge_fused', 'v19_a', 'v19_b')


class VOCJoint(Dataset):
    """VOC2012 paired image/mask, identical central square crop on train and val.

    No augmentation beyond fixed crop so A/B and RGB metric share identical images.
    Labels are class indices 0..20, 255=void, never RGB-interpolated.
    """
    def __init__(self, root, split, size, download=False, limit=0):
        from torchvision.datasets import VOCSegmentation
        self.data = VOCSegmentation(root=str(root), year='2012', image_set=split,
                                    download=download)
        self.size = size
        self.split = split
        self.n = len(self.data) if not limit else min(limit, len(self.data))
    def __len__(self): return self.n
    def __getitem__(self, idx):
        im, mask = self.data[idx]
        im, mask = segutil.crop_pair(im.convert('RGB'), mask, self.size)
        srgb = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2,0,1)/255.
        linear = v18.v15.srgb_to_linear(srgb)
        labels = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return linear, labels, idx


def load_seg_model(args, device):
    model, _ = segutil.get_segmentor(device, args.segmentor)
    model.requires_grad_(False)
    model.eval()
    return model


def seg_logits(model, rgb_srgb, input_size):
    """Differentiable, fixed preprocessing on both training and evaluation.

    Unlike torchvision weights.transforms() this keeps training compute bounded;
    GT and every reconstructed image use IDENTICAL normalization/resizing.
    """
    image = F.interpolate(rgb_srgb.clamp(0,1),size=(input_size,input_size),
                          mode='bilinear',align_corners=False)
    mu = image.new_tensor([.485,.456,.406])[None,:,None,None]
    std = image.new_tensor([.229,.224,.225])[None,:,None,None]
    logits = model((image-mu)/std)['out']
    return F.interpolate(logits.float(),size=rgb_srgb.shape[-2:],mode='bilinear',align_corners=False)


def differentiable_bridge_sample(net, z0, feats, budget, steps):
    """Same midpoint integrator as V18 inference, but WITH gradients.

    This is intentionally only called every --seg-every batches in V19-B.
    Critically, the segmentation branch NEVER sees the GT latent z1.
    """
    if steps < 1:raise ValueError('seg-train-steps must be >= 1')
    z=z0
    dt=1.0/steps
    for j in range(steps):
        t=j*dt
        v1=net(z,z0,feats,t,budget)
        v2=net(z+dt*.5*v1,z0,feats,t+dt*.5,budget)
        z=z+dt*v2
    return z


def get_autocast(args, device):
    if args.amp == 'none':return contextlib.nullcontext()
    if device.type != 'cuda':raise RuntimeError('AMP requires CUDA')
    if args.amp == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unsupported on this GPU; use --amp fp16 or none')
    dtype = torch.float16 if args.amp=='fp16' else torch.bfloat16
    return torch.autocast(device_type='cuda',dtype=dtype)


def grad_scaler(enabled):
    try:return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError,TypeError):return torch.cuda.amp.GradScaler(enabled=enabled)


def use_v18(args, device):
    cfg = argparse.Namespace(**vars(args));cfg.out=args.v18_out
    return v18.load_bridge(cfg, device)


def load_arm(args, device, arm, checkpoint='best'):
    cfg=argparse.Namespace(**vars(args));cfg.out=args.out/arm.upper()
    file=cfg.out/('bridge_best.pt' if checkpoint=='best' else 'bridge_last.pt')
    if not file.is_file():raise FileNotFoundError(f'V19-{arm.upper()} checkpoint missing: {file}')
    if checkpoint=='best':return v18.load_bridge(cfg,device)
    ck=torch.load(file,map_location='cpu',weights_only=False)
    for key in ('size','cycles','bg','mode','seed','vae_model','val_count'):
        if ck['config'][key]!=getattr(args,key):raise ValueError(f'{key} checkpoint mismatch')
    model=v18.Bridge(ck['config']['base'],ck['config']['latent_channels']).to(device)
    model.load_state_dict(ck['model'],strict=True)
    return model.eval()


def rgb_predict(net, vae, factor, adapter, feats, budget, args):
    y=adapter(feats).clamp(0,1)
    z0=v18.vae_encode(vae,v18.y_to_srgb3(y),factor)
    z=v18.bridge_sample(net,z0,feats,budget,steps=args.steps,
                        stochastic_start=args.sample_noise)
    pred=v18.vae_decode(vae,z,factor)
    return pred,y


@torch.no_grad()
def validate(net, vae, adapter, factor, ds, args, device, segmentor=None):
    net.eval()
    budget_psnr={}; seg_cms={}
    n=min(args.val_images,len(ds))
    for b in args.budgets:
        psnr=[]
        cm=np.zeros((21,21),dtype=np.int64)
        for idx in range(n):
            gt, mask, _=ds[idx]
            gt=gt[None].to(device)
            v18.v15.seed_photons(args,idx,b,salt=900000)
            feats,_,_,_,_=v18.v15.make_features(gt,b,args)
            pred,_=rgb_predict(net,vae,factor,adapter,feats,b,args)
            psnr.append(float(v18.v15.psnr_srgb(v18.v15.srgb_to_linear(pred),gt)[0]))
            if segmentor is not None and idx<args.val_seg_images:
                logits=seg_logits(segmentor,pred,args.seg_input_size)
                segutil.add_cm(cm,logits.argmax(1)[0].cpu().numpy(),mask.numpy())
        budget_psnr[b]=float(np.mean(psnr))
        if cm.sum():seg_cms[b]=segutil.cm_score(cm)['miou_foreground_present']
    net.train()
    return float(np.mean(list(budget_psnr.values()))),budget_psnr,seg_cms


def train(args):
    if args.arm not in ('A','B'):raise ValueError('Use --arm A or --arm B')
    if args.seg_weight<0 or args.kd_weight<0:raise ValueError('Loss weights must be nonnegative')
    device=v18.device_for(args)
    v18.seed_all(args.seed)
    init=args.v18_out/'bridge_best.pt'
    if not init.is_file():raise FileNotFoundError(f'Missing V18 checkpoint: {init}')
    train_ds=VOCJoint(args.voc_root,'train',args.size,args.download_voc,args.train_limit)
    val_ds=VOCJoint(args.voc_root,'val',args.size,args.download_voc)
    if set(map(str,train_ds.data.images))&set(map(str,val_ds.data.images)):
        raise RuntimeError('VOC train/val image leakage')
    print(f'VOC train={len(train_ds)} val={len(val_ds)}; arm={args.arm}',flush=True)
    vae,factor=v18.read_vae(args,device)
    adapter=v18.adapter_load(args,device)
    source=use_v18(args,device)
    source_ck=torch.load(init,map_location='cpu',weights_only=False)
    v18_cfg=source_ck['config']
    net=v18.Bridge(v18_cfg['base'],v18_cfg['latent_channels']).to(device)
    net.load_state_dict(source.state_dict(),strict=True)
    del source
    segmentor=load_seg_model(args,device) if args.arm=='B' else None
    optim=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=grad_scaler(args.amp=='fp16')
    dest=args.out/args.arm
    dest.mkdir(parents=True,exist_ok=True)
    # V18 loader compatibility is intentional: V19 saves exact V18 model format.
    config={k:v18_cfg[k] for k in ('mode','cycles','bg','size','seed','val_count','vae_model','base','latent_channels','noise_sigma','budgets')}
    (dest/'train_config.json').write_text(json.dumps({**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'v18_initial_checkpoint':str(init)},indent=2),encoding='utf8')
    writer=None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(dest/'tb'))
    best=-float('inf');hist=[];global_step=0
    try:
        for epoch in range(args.epochs):
            # Identical A/B stochastic schedule. Different DeepLab loading MUST NOT
            # affect photon seeds, flow time draws, or data order.
            v18.seed_all(args.seed+epoch)
            net.train();tic=time.perf_counter();n=0
            totals={k:0. for k in ('loss','flow','rgb','luma','photon','seg_ce','kd')}
            loader=DataLoader(train_ds,batch_size=args.batch,shuffle=True,
                              num_workers=args.workers,pin_memory=device.type=='cuda',
                              generator=torch.Generator().manual_seed(args.seed+100000+epoch))
            for idx,(gt,mask,_) in enumerate(loader):
                if args.max_batches_per_epoch and idx>=args.max_batches_per_epoch:break
                gt=gt.to(device,non_blocking=True);mask=mask.to(device,non_blocking=True)
                budget=args.budgets[(epoch*101+idx)%len(args.budgets)]
                v18.v15.seed_photons(args,idx+100000*epoch,budget,salt=400000)
                feats,k,_,_,_=v18.v15.make_features(gt,budget,args)
                do_seg=(segmentor is not None and idx%args.seg_every==0 and (mask!=255).any().item())
                with get_autocast(args,device):
                    with torch.no_grad():
                        y=adapter(feats)
                        z0=v18.vae_encode(vae,v18.y_to_srgb3(y),factor)
                        z1=v18.vae_encode(vae,v18.v15.linear_to_srgb(gt),factor)
                    flow,zt,vt,t,eps=v18.bridge_flow_loss(net,z0,z1,feats,budget,args.noise_sigma)
                    # Same differentiable one-step endpoint proxy as V18 training.
                    ts=t[:,None,None,None]
                    correction=args.noise_sigma*(torch.sin(math.pi*ts)+(1-ts)*math.pi*torch.cos(math.pi*ts))*eps
                    z_pred=zt+(1-ts)*vt-correction
                    srgb_pred=v18.vae_decode(vae,z_pred,factor)
                    srgb_gt=v18.v15.linear_to_srgb(gt)
                    rgb_loss=F.l1_loss(srgb_pred,srgb_gt)
                    pred_y=v18.v15.luminance(v18.v15.srgb_to_linear(srgb_pred))
                    lum_loss=F.l1_loss(pred_y,v18.v15.luminance(gt))
                    ph_loss=v18.photon_nll_linear_y(pred_y,k,budget,args)
                    loss=flow+args.lambda_rgb*rgb_loss+args.lambda_luma*lum_loss+args.lambda_ph*ph_loss
                    ce=loss.new_zeros(());kd=loss.new_zeros(())
                    if do_seg:
                        # Segmentation is supervised on the EXACT inference-style
                        # path from measured photon -> z0 -> Bridge, not on the
                        # GT-latent-contaminated one-step training proxy above.
                        z_seg=differentiable_bridge_sample(net,z0,feats,budget,args.seg_train_steps)
                        rgb_seg=v18.vae_decode(vae,z_seg,factor)
                        # Only bridge gradients; segmentor parameters are frozen,
                        # but no no_grad() around the student DeepLab forward!
                        logits=seg_logits(segmentor,rgb_seg,args.seg_input_size)
                        ce=F.cross_entropy(logits,mask,ignore_index=255)
                        loss=loss+args.seg_weight*args.seg_every*ce
                        if args.kd_weight>0:
                            with torch.no_grad():teacher=seg_logits(segmentor,srgb_gt,args.seg_input_size)
                            temp=args.kd_temperature
                            valid=(mask!=255)
                            dist=F.kl_div(F.log_softmax(logits.float()/temp,dim=1),
                                          F.softmax(teacher.float()/temp,dim=1),reduction='none').sum(1)
                            kd=dist[valid].mean()*temp*temp
                            loss=loss+args.kd_weight*args.seg_every*kd
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss epoch={epoch+1}, batch={idx+1}')
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if scaler.is_enabled():scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(net.parameters(),args.grad_clip)
                scaler.step(optim);scaler.update()
                values={'loss':loss,'flow':flow,'rgb':rgb_loss,'luma':lum_loss,'photon':ph_loss,'seg_ce':ce,'kd':kd}
                for name,value in values.items():totals[name]+=float(value.detach())*len(gt)
                n+=len(gt);global_step+=1
                if writer and idx%20==0:
                    for name,value in values.items():writer.add_scalar('batch/'+name,float(value.detach()),global_step)
                if args.log_every and ((idx+1)%args.log_every==0 or idx==0):
                    print(f'arm={args.arm} epoch={epoch+1}/{args.epochs} batch={idx+1}/{len(loader)} '
                          f'loss={float(loss.detach()):.4f} seg={float(ce.detach()):.3f} '
                          f'{n/max(time.perf_counter()-tic,1e-6):.1f} img/s',flush=True)
            if n==0:raise RuntimeError('No VOC training batches')
            score=float('nan');budget_scores={};seg_scores={}
            if (epoch+1)%args.val_every==0 or epoch+1==args.epochs:
                score,budget_scores,seg_scores=validate(net,vae,adapter,factor,val_ds,args,device,segmentor)
                if score>best:
                    best=score
                    torch.save({'model':net.state_dict(),'config':config,'epoch':epoch+1,
                                'val_psnr':score,'best_val_psnr':best,'arm':args.arm},dest/'bridge_best.pt')
            torch.save({'model':net.state_dict(),'optimizer':optim.state_dict(),
                        'scaler':scaler.state_dict() if scaler.is_enabled() else None,
                        'config':config,'epoch':epoch+1,'val_psnr':score,'best_val_psnr':best,
                        'arm':args.arm,'global_step':global_step},dest/'bridge_last.pt')
            row={'epoch':epoch+1,'arm':args.arm,'val_psnr':score,'val_fg_miou_mean':float(np.mean(list(seg_scores.values()))) if seg_scores else float('nan')}
            row.update({f'train_{k}':v/n for k,v in totals.items()})
            row.update({f'val_psnr_B{b:g}':budget_scores.get(b,float('nan')) for b in args.budgets})
            row.update({f'val_fg_miou_B{b:g}':seg_scores.get(b,float('nan')) for b in args.budgets})
            hist.append(row);v18.write_csv(dest/'history.csv',hist)
            print(f'ARM {args.arm} epoch={epoch+1}/{args.epochs} val_PSNR={score:.3f} best={best:.3f} val_fg_mIoU={row["val_fg_miou_mean"]:.3f}',flush=True)
            if writer:
                for key,value in row.items():
                    if key.startswith(('train_','val_')) and np.isfinite(value):writer.add_scalar('epoch/'+key,value,epoch+1)
                writer.flush()
    finally:
        if writer:writer.close()
    print('Saved',dest/'bridge_best.pt',dest/'bridge_last.pt',flush=True)


@torch.no_grad()
def load_for_eval(args,device):
    methods=args.methods
    legacy_methods=[m for m in methods if m in v18.ALL]
    vae,factor=v18.read_vae(args,device)
    adapter,legacy,colorizer=v18.load_models(args,device,legacy_methods)
    baseline=use_v18(args,device) if set(legacy_methods)&{'bridge','bridge_fused'} else None
    arms={}
    for code in ('a','b'):
        if any(m.startswith('v19_'+code) for m in methods):
            arms[code]=load_arm(args,device,code,args.checkpoint)
    return vae,factor,adapter,legacy,colorizer,baseline,arms


@torch.no_grad()
def all_outputs(gt,idx,budget,args,device,models):
    vae,factor,adapter,legacy,colorizer,baseline,arms=models
    old=[m for m in args.methods if m in v18.ALL]
    # V18 observations generated exactly once, using its evaluation seed.
    outputs,inter=v18.predictions_for_sample(gt,idx,budget,args,device,vae,factor,adapter,
                                              baseline,legacy,colorizer,old)
    if arms:
        v18.v15.seed_photons(args,idx,budget,salt=300000)
        feats,_,_,_,_=v18.v15.make_features(gt,budget,args)
        for code,net in arms.items():
            srgb,y=rgb_predict(net,vae,factor,adapter,feats,budget,args)
            if f'v19_{code}' in args.methods:outputs[f'v19_{code}']=v18.v15.srgb_to_linear(srgb)
            if f'v19_{code}_fused' in args.methods:
                outputs[f'v19_{code}_fused']=v18.v15.srgb_to_linear(v18.fit_luminance(srgb,y,args.luma_fusion))
    return outputs,inter


def eval_rgb(args):
    device=v18.device_for(args)
    dataset=v18.v15.RGBImages(args.data,args.split,args.size,args.seed,args.val_count)
    models=load_for_eval(args,device)
    dest=args.out/f'eval_rgb_{args.split}'
    dest.mkdir(parents=True,exist_ok=True)
    rows=[];n=len(dataset) if args.eval_images==0 else min(args.eval_images,len(dataset))
    with torch.no_grad():
        for idx in range(n):
            gt,path=dataset[idx];gt=gt[None].to(device)
            per=[]
            for b in args.budgets:
                outputs,inter=all_outputs(gt,idx,b,args,device,models)
                meta={'scene':path,'budget_total':b}
                for m in args.methods:
                    rows.append({**meta,'method':m,'is_oracle':int(m in ('ddcolor_oracle','vae_oracle')),
                                 **v18.v16.compute_metrics(outputs[m],gt)})
                if idx<args.gallery_n:
                    k,est,bf,y=inter
                    cols=[('Counts',v18.v16.to_pil_srgb((k/3).clamp(0,1))),
                          ('MLE+BF',v18.v16.to_pil_srgb(bf.clamp(0,1)))]
                    cols += [(m,v18.v16.to_pil_srgb(outputs[m])) for m in args.methods]
                    cols += [('RGB GT',v18.v16.to_pil_srgb(gt))]
                    per.append((b,cols))
            if per:v18.v16.save_gallery(per,args.size,dest/f'{idx:04d}.png')
            if (idx+1)%10==0 or idx+1==n:print('RGB',idx+1,'/',n,flush=True)
    v18.write_csv(dest/'metrics.csv',rows)
    summaries=[]
    for b in args.budgets:
        for name in args.methods:
            rr=[r for r in rows if r['budget_total']==b and r['method']==name]
            row={'budget_total':b,'method':name,'images':len(rr)}
            for key in ('psnr_srgb','ssim_srgb','lab_ab_error','delta_e76','linear_luma_mse'):
                row[key]=float(np.mean([r[key] for r in rr]))
            summaries.append(row)
            print(f'B={b:g} {name:16s} PSNR={row["psnr_srgb"]:.3f} SSIM={row["ssim_srgb"]:.3f} ab={row["lab_ab_error"]:.2f}',flush=True)
    v18.write_csv(dest/'summary.csv',summaries)
    print('Saved',dest,flush=True)


def eval_seg(args):
    device=v18.device_for(args)
    ds=VOCJoint(args.voc_root,'val',args.size,args.download_voc)
    models=load_for_eval(args,device)
    segmentor=load_seg_model(args,device)
    n=len(ds) if args.eval_images==0 else min(args.eval_images,len(ds))
    dest=args.seg_out or (args.out/'eval_seg_voc')
    dest.mkdir(parents=True,exist_ok=True)
    cms={(m,b):np.zeros((21,21),dtype=np.int64) for m in ['GT']+list(args.methods) for b in args.budgets}
    rows=[];gallery={b:[] for b in args.budgets}
    show=[m for m in args.seg_gallery_methods if m in args.methods]
    if not show:show=[m for m in args.methods if m not in ('ddcolor_oracle','vae_oracle')][:6]
    with torch.no_grad():
        for idx in range(n):
            gt,mask,_=ds[idx];gt=gt[None].to(device)
            rgb_gt=v18.v15.linear_to_srgb(gt)
            gt_logits=seg_logits(segmentor,rgb_gt,args.seg_input_size)
            ref=gt_logits.argmax(1)[0].cpu().numpy()
            true=mask.numpy()
            for b in args.budgets:
                outputs,_=all_outputs(gt,idx,b,args,device,models)
                preds={'GT':ref}
                for m in args.methods:
                    out_srgb=v18.v15.linear_to_srgb(outputs[m])
                    logits=seg_logits(segmentor,out_srgb,args.seg_input_size)
                    preds[m]=logits.argmax(1)[0].cpu().numpy()
                for m,p in preds.items():
                    local=np.zeros((21,21),dtype=np.int64)
                    segutil.add_cm(local,p,true)
                    cms[(m,b)]+=local
                    rows.append({'scene_id':idx,'budget_total':b,'method':m,**segutil.cm_score(local)})
                if len(gallery[b])<args.seg_gallery_n:
                    sample={'index':idx,'budget':b,'gt_rgb':segutil.linear_rgb_to_pil(gt),
                            'gt_mask':true.copy(),'gt_seg':ref.copy(),
                            'rgb':{m:segutil.linear_rgb_to_pil(outputs[m]) for m in show},
                            'seg':{m:preds[m].copy() for m in show}}
                    gallery[b].append(sample)
            if (idx+1)%10==0 or idx+1==n:print('VOC seg',idx+1,'/',n,flush=True)
    v18.write_csv(dest/'per_image.csv',rows)
    summaries=[]
    for b in args.budgets:
        for m in ['GT']+list(args.methods):
            sc=segutil.cm_score(cms[(m,b)])
            summaries.append({'budget_total':b,'method':m,'images':n,**sc})
            print(f'B={b:g} {m:16s} mIoU={sc["miou_present_21"]:.4f} fg_mIoU={sc["miou_foreground_present"]:.4f} acc={sc["pixel_accuracy"]:.4f}',flush=True)
        if gallery[b]:segutil.save_segmentation_gallery(dest/f'seg_gallery_B{b:g}.png',gallery[b],show,tile=args.seg_gallery_tile)
    v18.write_csv(dest/'summary.csv',summaries)
    np.savez_compressed(dest/'confusions.npz',**{f'{m}_B{b:g}':cm for (m,b),cm in cms.items()})
    (dest/'config.json').write_text(json.dumps({**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'preprocessing':f'bilinear_resize_{args.seg_input_size}+imagenet_norm, logits->VOC mask native crop'},indent=2),encoding='utf8')
    print('Saved',dest,flush=True)


def check(args):
    device=v18.device_for(args)
    print('V18 checkpoint:',args.v18_out/'bridge_best.pt')
    if not (args.v18_out/'bridge_best.pt').is_file():raise FileNotFoundError('V18 bridge_best.pt missing')
    ds=VOCJoint(args.voc_root,'train',args.size,args.download_voc,args.train_limit)
    dv=VOCJoint(args.voc_root,'val',args.size,args.download_voc)
    print('VOC train:',len(ds),'VOC val:',len(dv),'example shapes:',[x.shape if torch.is_tensor(x) else x for x in ds[0]],flush=True)
    print('Checking V18 architecture and checkpoint...',flush=True)
    use_v18(args,device)
    print('Check PASSED',flush=True)


def main():
    p=v18.build_parser()
    for action in p._actions:
        if action.dest=='action':action.choices=('check','train','eval_rgb','eval_seg')
        if action.dest=='methods':action.choices=list(v18.ALL)+list(V19_NAMES);action.default=list(DEFAULT_METHODS)
    p.add_argument('--arm',choices=['A','B'],default='A')
    p.add_argument('--v18-out',type=Path,default=Path('runs/v18_bridge_fast'))
    p.add_argument('--voc-root',type=Path,default=Path('data/VOC'))
    p.add_argument('--download-voc',action='store_true')
    p.add_argument('--train-limit',type=int,default=0,help='VOC train images; 0=all, >0 for quick testing')
    p.add_argument('--seg-out',type=Path,default=None)
    p.add_argument('--segmentor',choices=['deeplab_mobilenet','deeplab_resnet50'],default='deeplab_mobilenet')
    p.add_argument('--seg-input-size',type=int,default=256,help='Frozen segmentor input pixels; 520 is closer to its pretrained transform but slower')
    p.add_argument('--seg-every',type=int,default=4,help='Calculate semantic loss every N batches in B; multiplied by N for expected weight')
    p.add_argument('--seg-train-steps',type=int,default=2,help='Differentiable midpoint sampling steps for B segmentation supervision; evaluation uses --steps')
    p.add_argument('--seg-weight',type=float,default=.03)
    p.add_argument('--kd-weight',type=float,default=.005)
    p.add_argument('--kd-temperature',type=float,default=2.)
    p.add_argument('--val-seg-images',type=int,default=3)
    p.add_argument('--checkpoint',choices=['best','last'],default='best')
    p.add_argument('--seg-gallery-n',type=int,default=6)
    p.add_argument('--seg-gallery-tile',type=int,default=176)
    p.add_argument('--seg-gallery-methods',nargs='+',choices=list(v18.ALL)+list(V19_NAMES),
                   default=['raw','full','bridge','v19_a','v19_b'])
    args=p.parse_args()
    if args.size%32 or args.size<32:p.error('--size must be a positive multiple of 32')
    if args.seg_every<1 or args.seg_input_size<32 or args.seg_train_steps<1:p.error('invalid segmentation interval, steps or input size')
    if args.seg_gallery_n<0 or args.val_images<1:p.error('gallery must be nonnegative, val-images positive')
    if args.max_batches_per_epoch<0 or args.train_limit<0:p.error('limits cannot be negative')
    if args.action=='check':check(args)
    elif args.action=='train':train(args)
    elif args.action=='eval_rgb':eval_rgb(args)
    else:eval_seg(args)

if __name__=='__main__':main()
