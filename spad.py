#!/usr/bin/env python3
"""Consolidated SPAD research CLI: train, resume, infer, RGB/VOC eval and plotting.

Two source files: spad.py + spad_core.py. Existing checkpoints remain unchanged.
Trains V15/V17 and optional V18 Bridge; original checkpoint schema is preserved.
Training defaults to a new output directory and never overwrites original runs.

Examples:
  python spad.py check
  python spad.py rgb --model full --split test --eval-images 20
  python spad.py rgb --models raw mle full adapter_ddcolor bridge_fused
  python spad.py seg --models full adapter_ddcolor --eval-images 20
  python spad.py clean                  # dry-run
  python spad.py clean --apply          # zip archive, verify, then remove legacy root files
"""
import argparse
import csv
import json
import math
import random
import time
import contextlib
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import spad_core as C

METHODS = ('raw','mle','full','adapter_ddcolor','ddcolor_mle_bf',
           'ddcolor_full_luma','ddcolor_oracle','bridge','bridge_fused')
V15 = {'raw','mle','full'}
DDC = {'adapter_ddcolor','ddcolor_mle_bf','ddcolor_full_luma','ddcolor_oracle'}
BRIDGE = {'bridge','bridge_fused'}


def choose_models(args):
    models = list(args.models or []) + list(args.one_model or [])
    models = models or (['full'] if args.action in ('train','infer') else ['raw','mle','full','adapter_ddcolor'])
    if len(models) != len(set(models)):
        raise ValueError('Duplicate --models or --model entries')
    if args.action != 'train' and 'adapter' in models:
        raise ValueError('adapter is a luminance-only TRAIN model; choose adapter_ddcolor for RGB/seg/infer')
    return models


def config_for_run(args,models):
    cfg={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    cfg['models_resolved']=models
    cfg['oracle_methods']=[m for m in models if m=='ddcolor_oracle']
    return cfg


def save_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


class Predictor:
    def __init__(self,args,models):
        self.args=args;self.methods=models
        self.device=C.get_device(args.device)
        requested=set(models)
        v15_names=(requested & V15) | ({'full'} if 'ddcolor_full_luma' in requested else set())
        self.direct={name:C.load_v15(args,self.device,name) for name in sorted(v15_names)}
        need_adapter=bool(requested & ({'adapter_ddcolor'} | BRIDGE))
        self.adapter=C.load_adapter(args,self.device) if need_adapter else None
        self.ddcolor=C.load_ddcolor(args,self.device)[0] if requested & DDC else None
        self.vae=None;self.scale=None;self.bridge=None
        if requested & BRIDGE:
            self.bridge=C.load_bridge(args,self.device)
            self.vae,self.scale=C.read_vae(args,self.device)
        print(f'Models ready: {models} on {self.device}',flush=True)

    @torch.inference_mode()
    def predict(self,gt,index,budget):
        """Simulate photon counts only for RGB datasets with ground truth."""
        C.seed_photons(self.args,index,budget,salt=300000)
        features,k,est,var,bf=C.make_features(gt,budget,self.args)
        return self.predict_features(features,k,est,var,bf,budget,gt=gt)

    @torch.inference_mode()
    def predict_features(self,features,k,est,var,bf,budget,gt=None):
        """Predict from observed counts, with GT used ONLY by the explicit Oracle."""
        a=self.args
        preds={}
        for name,net in self.direct.items():
            preds[name]=net(C.make_variant(features,name))
        if self.adapter is not None:
            y=self.adapter(features).clamp(0,1)
        else:y=None
        if 'adapter_ddcolor' in self.methods:
            preds['adapter_ddcolor']=C.ddcolorize(self.ddcolor,y,self.device)
        if 'ddcolor_mle_bf' in self.methods:
            preds['ddcolor_mle_bf']=C.ddcolorize(self.ddcolor,bf.clamp(0,1),self.device)
        if 'ddcolor_full_luma' in self.methods:
            preds['ddcolor_full_luma']=C.ddcolorize(self.ddcolor,C.luminance(preds['full']),self.device)
        if 'ddcolor_oracle' in self.methods:
            if gt is None:raise ValueError('Oracle requires RGB ground truth and is not deployable')
            preds['ddcolor_oracle']=C.ddcolorize(self.ddcolor,C.luminance(gt),self.device)
        if 'bridge' in self.methods:
            preds['bridge']=C.bridge_predict(self.vae,self.scale,self.bridge,y,features,budget,a.steps,0.0)
        if 'bridge_fused' in self.methods:
            preds['bridge_fused']=C.bridge_predict(self.vae,self.scale,self.bridge,y,features,budget,a.steps,a.luma_fusion)
        return {m:preds[m] for m in self.methods},(k,est,bf,y)


def do_check(args):
    device=C.get_device(args.device)
    paths = [args.v15_out / 'full_best.pt',args.v17_out / 'luma_adapter_best.pt']
    for p in paths:
        if not p.is_file():raise FileNotFoundError(f'Required checkpoint missing: {p}')
    C.load_v15(args,device,'full')
    C.load_adapter(args,device)
    if not (args.ddcolor_repo/'ddcolor'/'__init__.py').is_file():
        print(f'WARNING: official DDColor source missing: {args.ddcolor_repo}')
    print('CHECK PASSED: V15 Full and V17 Adapter checkpoint architectures/configurations match.')
    print('No pretrained DDColor download was attempted during check.')


def do_rgb(args):
    methods=choose_models(args)
    if 'ddcolor_oracle' in methods:
        print('NOTE: ddcolor_oracle uses GT luminance; not deployable.',flush=True)
    ds=C.RGBImages(args.data,args.split,args.size,args.seed,args.val_count)
    predictor=Predictor(args,methods)
    n=len(ds) if args.eval_images==0 else min(len(ds),args.eval_images)
    if n<1:raise ValueError('No RGB images to evaluate')
    dest=args.out/('rgb_'+args.split);dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(n):
        gt,source=ds[i];gt=gt[None].to(predictor.device)
        gallery=[]
        for b in args.budgets:
            preds,(k,est,bf,y)=predictor.predict(gt,i,b)
            for m in methods:
                metrics=C.compute_metrics(preds[m],gt)
                rows.append({'scene':source,'image_index':i,'budget_total':b,'method':m,
                             'is_oracle':int(m=='ddcolor_oracle'),**metrics})
            if i<args.gallery_n:
                cols=[('Count',C.to_pil_srgb((k/3).clamp(0,1))),('MLE',C.to_pil_srgb(est.clamp(0,1))),
                      ('MLE+BF',C.to_pil_srgb(bf.clamp(0,1)))]
                if y is not None:cols.append(('Adapter Y',C.to_pil_srgb(y)))
                cols.extend((m,C.to_pil_srgb(preds[m])) for m in methods)
                cols.append(('GT RGB',C.to_pil_srgb(gt)))
                gallery.append((b,cols))
        if gallery:C.save_gallery(gallery,args.size,dest/f'{i:04d}.png')
        if (i+1)%10==0 or i+1==n:print(f'RGB: {i+1}/{n}',flush=True)
    summary=[]
    for b in args.budgets:
        for m in methods:
            selected=[r for r in rows if r['budget_total']==b and r['method']==m]
            rec={'budget_total':b,'method':m,'images':len(selected),'is_oracle':int(m=='ddcolor_oracle')}
            for key in ('psnr_srgb','ssim_srgb','lab_ab_error','delta_e76','linear_luma_mse','linear_rgb_L1'):
                rec[key]=float(np.mean([r[key] for r in selected]))
            summary.append(rec)
            print(f'B={b:g}, {m:20s}, PSNR={rec["psnr_srgb"]:.3f}, SSIM={rec["ssim_srgb"]:.4f}')
    save_csv(dest/'metrics.csv',rows);save_csv(dest/'summary.csv',summary)
    (dest/'config.json').write_text(json.dumps(config_for_run(args,methods),indent=2),encoding='utf8')
    print('RGB results:',dest.resolve())


def voc_palette():
    colors=np.zeros((21,3),dtype=np.uint8)
    for label in range(21):
        v=label
        for bit in range(8):
            for chan in range(3):colors[label,chan]|=((v>>chan)&1) << (7-bit)
            v >>= 3
    return colors

COLORS=voc_palette()

def color_mask(mask):
    a=np.asarray(mask,dtype=np.int64)
    colored=np.full((*a.shape,3),255,dtype=np.uint8)
    good=(a>=0)&(a<21)
    colored[good]=COLORS[a[good]]
    return Image.fromarray(colored,'RGB')


def crop_voc(img,mask,size):
    w,h=img.size;s=min(w,h);left=(w-s)//2;top=(h-s)//2
    box=(left,top,left+s,top+s)
    return (img.convert('RGB').crop(box).resize((size,size),Image.Resampling.BICUBIC),
            mask.crop(box).resize((size,size),Image.Resampling.NEAREST))


def load_segmentor(device,kind):
    from torchvision.models.segmentation import (deeplabv3_resnet50,DeepLabV3_ResNet50_Weights,
        deeplabv3_mobilenet_v3_large,DeepLabV3_MobileNet_V3_Large_Weights)
    if kind=='resnet50':
        weights=DeepLabV3_ResNet50_Weights.DEFAULT
        net=deeplabv3_resnet50(weights=weights)
    else:
        weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        net=deeplabv3_mobilenet_v3_large(weights=weights)
    net=net.to(device).eval().requires_grad_(False)
    return net,weights.transforms()


@torch.inference_mode()
def predict_mask(net,preprocess,srgb,target_size):
    x=preprocess(srgb[0].float().clamp(0,1)).unsqueeze(0)
    logits=net(x)['out']
    return F.interpolate(logits.float(),size=target_size,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)


def update_cm(cm,pred,gt):
    valid=(gt!=255)&(gt>=0)&(gt<21)&(pred>=0)&(pred<21)
    counts=np.bincount(gt[valid].astype(np.int64)*21+pred[valid].astype(np.int64),minlength=441)
    cm+=counts.reshape(21,21)


def cm_stats(cm,classes=None):
    gt_present=cm.sum(1)>0 if classes is None else classes
    tp=np.diag(cm).astype(float);unions=cm.sum(1)+cm.sum(0)-tp
    iou=tp/np.maximum(1,unions)
    selected=gt_present & (unions>0)
    fg=selected.copy();fg[0]=False
    return {'miou_gt_present':float(iou[selected].mean()) if selected.any() else float('nan'),
            'foreground_miou_gt_present':float(iou[fg].mean()) if fg.any() else float('nan'),
            'pixel_accuracy':float(tp.sum()/max(cm.sum(),1))}


def image_gallery(path,rows,methods,tile):
    # Per scene two rows: top RGB images, bottom semantic masks.
    if not rows:return
    headers=['VOC GT','Sparse intensity','DeepLab GT']+methods
    margin=12;gap=8;head=55;block=tile*2+43
    w=2*margin+len(headers)*(tile+gap);h=head+len(rows)*block+20
    canvas=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(canvas)
    draw.text((margin,8),f'VOC 2012 B={rows[0]["budget"]:g} | Official DeepLab | RGB / masks',fill='black')
    for j,m in enumerate(headers):draw.text((margin+j*(tile+gap),30),m[:21],fill='black')
    for i,r in enumerate(rows):
        yy=head+i*block
        draw.text((margin,yy),f'VOC val index {r["index"]}',fill='black')
        yy+=20
        rgb=[r['gt_rgb'],r['counts_display'],r['gt_rgb']]+[r['rgb'][m] for m in methods]
        seg=[color_mask(r['gt_mask']),None,color_mask(r['gt_pred'])]+[color_mask(r['masks'][m]) for m in methods]
        for j,(a,b) in enumerate(zip(rgb,seg)):
            x=margin+j*(tile+gap)
            canvas.paste(a.resize((tile,tile),Image.Resampling.BICUBIC),(x,yy))
            if b is not None:
                canvas.paste(b.resize((tile,tile),Image.Resampling.NEAREST),(x,yy+tile+3))
            else:
                # The shared sparse observation is not a segmentation prediction.
                draw.rectangle((x,yy+tile+3,x+tile-1,yy+2*tile+2),fill=(247,248,250))
                draw.text((x+9,yy+tile+tile//2-4),'Input only',fill=(90,95,106))
    canvas.save(path)


def do_seg(args):
    from torchvision.datasets import VOCSegmentation
    methods=choose_models(args)
    if 'ddcolor_oracle' in methods:print('NOTE: Oracle uses GT luminance.',flush=True)
    device=C.get_device(args.device)
    ds=VOCSegmentation(root=str(args.voc_root),year='2012',image_set='val',download=False)
    if args.skip_images>=len(ds):raise ValueError('--skip-images exceeds VOC val length')
    ids=list(range(args.skip_images,len(ds)))
    if args.eval_images>0:ids=ids[:args.eval_images]
    predictor=Predictor(args,methods)
    segmentor,preprocess=load_segmentor(device,args.segmentor)
    dest=args.out/'seg_voc';dest.mkdir(parents=True,exist_ok=True)
    cms={(m,b):np.zeros((21,21),np.int64) for b in args.budgets for m in ['GT_crop']+methods}
    rows=[];galleries={b:[] for b in args.budgets}
    for j,index in enumerate(ids):
        original,msk=ds[index]
        crop,msk=crop_voc(original,msk,args.size)
        truth=np.asarray(msk,dtype=np.uint8)
        src=torch.from_numpy(np.asarray(crop,dtype=np.float32).copy()).permute(2,0,1)[None].to(device)/255
        gt=C.srgb_to_linear(src)
        pred_gt=predict_mask(segmentor,preprocess,src,truth.shape)
        present=np.bincount(truth[truth<21],minlength=21)>0
        for b in args.budgets:
            preds,(k,est,bf,y)=predictor.predict(gt,index,b)
            predicted={'GT_crop':pred_gt}
            for m in methods:
                srgb=C.linear_to_srgb(preds[m])
                predicted[m]=predict_mask(segmentor,preprocess,srgb,truth.shape)
            for m,p in predicted.items():
                cm=np.zeros((21,21),np.int64);update_cm(cm,p,truth);cms[(m,b)]+=cm
                rows.append({'voc_index':index,'budget_total':b,'method':m,**cm_stats(cm,present)})
            if len(galleries[b])<args.gallery_n:
                galleries[b].append({'index':index,'budget':b,'gt_rgb':crop,'gt_mask':truth,
                    'gt_pred':pred_gt,
                    'counts_display':C.to_pil_srgb((k/3).clamp(0,1)),
                    'rgb':{m:C.to_pil_srgb(preds[m]) for m in methods},
                    'masks':{m:predicted[m] for m in methods}})
        if (j+1)%5==0 or j+1==len(ids):print(f'VOC: {j+1}/{len(ids)}',flush=True)
    gtc=(cms[('GT_crop',args.budgets[0])].sum(1)>0)
    summary=[]
    for b in args.budgets:
        for m in ['GT_crop']+methods:
            r={'budget_total':b,'method':m,'images':len(ids),**cm_stats(cms[(m,b)],gtc)}
            summary.append(r)
            print(f'B={b:g}, {m:20s}, mIoU={r["miou_gt_present"]:.4f}, FG-mIoU={r["foreground_miou_gt_present"]:.4f}')
        image_gallery(dest/f'seg_gallery_B{b:g}.png',galleries[b],methods,args.gallery_tile)
    save_csv(dest/'per_image.csv',rows);save_csv(dest/'summary.csv',summary)
    (dest/'config.json').write_text(json.dumps(config_for_run(args,methods),indent=2),encoding='utf8')
    print('VOC results:',dest.resolve())



def _write_training_checkpoint(path,model,optimizer,epoch,args,train_model,best,scaler=None):
    """Best models retain exactly the historical V15/V17/V18 loader layout."""
    meta={k:getattr(args,k) for k in ('mode','cycles','bg','size','seed','val_count',
             'bf_radius','bf_sigma','bf_knoise','bf_prec_cap')}
    if train_model in V15:
        ck={**meta,'state_dict':model.state_dict(),'epoch':epoch,'base':args.base,
            'variant':train_model,'budgets':list(args.budgets),'val_score':best}
    elif train_model=='adapter':
        ck={**meta,'state_dict':model.state_dict(),'epoch':epoch,'base':args.base,
            'budgets':list(args.budgets),'lambda_edge':args.lambda_edge,
            'lambda_ph':args.lambda_ph,'val_score':best}
    else:
        ck={'model':model.state_dict(),'epoch':epoch,'val_psnr':-best,
            'config':{k:getattr(args,k) for k in ('mode','cycles','bg','size','seed',
                    'val_count','vae_model','noise_sigma','budgets')},
            'best_val_psnr':-best}
        ck['config'].update({'base':args.base,'latent_channels':model.latent_channels})
    return ck


def _train_paths(args,model_name):
    if model_name in V15:
        folder=args.train_out/'v15_64'; best=folder/f'{model_name}_best.pt';last=folder/f'{model_name}_last.pt'
    elif model_name=='adapter':
        folder=args.train_out/'v17_adapter';best=folder/'luma_adapter_best.pt';last=folder/'luma_adapter_last.pt'
    elif model_name=='bridge':
        folder=args.train_out/'v18_bridge';best=folder/'bridge_best.pt';last=folder/'bridge_last.pt'
    else:raise ValueError(f'Cannot train {model_name}; DDColor is frozen/pretrained.')
    return folder,best,last


def _make_training_model(name,args,device):
    if name in V15:return C.RGBUNet(args.base).to(device)
    if name=='adapter':return C.LumaAdapter(args.base).to(device)
    if name=='bridge':return C.Bridge(args.base).to(device)
    raise ValueError(f'Unsupported training model: {name}')


def _training_eval(model,name,valset,args,device,adapter=None,vae=None,factor=None,max_images=None):
    """Model selection uses FIXED val scenes/budgets and fixed independent photon seeds."""
    model.eval();scores=[];preview=[]
    with torch.no_grad():
        for i in range(min(args.val_images if max_images is None else max_images,len(valset))):
            gt,_=valset[i];gt=gt[None].to(device)
            for b in args.budgets:
                C.seed_photons(args,i,b,salt=200000)
                feats,k,est,var,bf=C.make_features(gt,b,args)
                if name in V15:
                    pred=model(C.make_variant(feats,name));score=F.l1_loss(pred,gt).item()
                    rendered=C.linear_to_srgb(pred)
                elif name=='adapter':
                    yp=model(feats);score=F.l1_loss(yp,C.luminance(gt)).item()
                    rendered=C.linear_to_srgb(yp).repeat(1,3,1,1)
                else:
                    yy=adapter(feats);z0=C.vae_encode(vae,C.y_to_srgb3(yy),factor)
                    z=C.bridge_sample(model,z0,feats,b,args.steps)
                    srgb=C.vae_decode(vae,z,factor)
                    score=-float(C.compute_metrics(C.srgb_to_linear(srgb),gt)['psnr_srgb'])
                    rendered=srgb
                scores.append(score)
                if i<args.tb_images and b in (args.budgets[0],args.budgets[-1]):
                    # Counts are visualized as count/3, NOT a calibrated intensity.
                    count_view=(k/3).clamp(0,1).repeat(1,3,1,1)
                    bf_view=C.linear_to_srgb(bf.clamp(0,1)).repeat(1,3,1,1)
                    strip=torch.cat([count_view[0],bf_view[0],rendered[0],C.linear_to_srgb(gt)[0]],dim=2)
                    preview.append((i,b,strip.detach().cpu()))
    return float(np.mean(scores)),preview


def _training_step(model,name,gt,budget,feats,k,args,adapter=None,vae=None,factor=None):
    if name in V15:
        return C.rgb_training_loss(model(C.make_variant(feats,name)),gt,k,budget,args)[0]
    if name=='adapter':
        return C.adapter_training_loss(model(feats),gt,k,budget,args)[0]
    with torch.no_grad():
        y=adapter(feats)
        z0=C.vae_encode(vae,C.y_to_srgb3(y),factor)
        z1=C.vae_encode(vae,C.linear_to_srgb(gt),factor)
    flow,zt,vt,t,eps=C.bridge_flow_loss(model,z0,z1,feats,budget,args.noise_sigma)
    loss=flow
    if args.lambda_rgb or args.lambda_y or args.lambda_ph:
        ts=t[:,None,None,None]
        correction=args.noise_sigma*(torch.sin(math.pi*ts)+(1-ts)*math.pi*torch.cos(math.pi*ts))*eps
        pred_z=zt+(1-ts)*vt-correction
        srgb=C.vae_decode(vae,pred_z,factor)
        if args.lambda_rgb:loss=loss+args.lambda_rgb*F.l1_loss(srgb,C.linear_to_srgb(gt))
        if args.lambda_y or args.lambda_ph:
            yp=C.luminance(C.srgb_to_linear(srgb))
            if args.lambda_y:loss=loss+args.lambda_y*F.l1_loss(yp,C.luminance(gt))
            if args.lambda_ph:loss=loss+args.lambda_ph*C.photon_nll(yp,k,budget,args)
    return loss


def _do_single_train(args,name):
    device=C.get_device(args.device)
    ds=C.RGBImages(args.data,'train',args.size,args.seed,args.val_count)
    dv=C.RGBImages(args.data,'val',args.size,args.seed,args.val_count)
    if set(ds.paths)&set(dv.paths):raise RuntimeError('Training/validation data leakage')
    folder,bestfile,lastfile=_train_paths(args,name);folder.mkdir(parents=True,exist_ok=True)
    if bestfile.exists() and not args.overwrite:
        if args.resume_from is None or args.resume_from.resolve()!=lastfile.resolve():
            raise FileExistsError(f'{bestfile} exists. For exact continuation use --resume-from {lastfile}; otherwise choose new --train-out or explicit --overwrite.')
    if args.resume_from and len(args.models or []) + len(args.one_model or []) > 1:
        raise ValueError('--resume-from requires exactly one --model')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    model=_make_training_model(name,args,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    amp_dtype={'none':None,'fp16':torch.float16,'bf16':torch.bfloat16}[args.amp]
    if device.type!='cuda' and amp_dtype is not None:
        raise ValueError('--amp fp16/bf16 requires CUDA')
    try:
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and args.amp=='fp16')
    except (AttributeError,TypeError):
        scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and args.amp=='fp16')
    adapter=vae=factor=None
    if name=='bridge':
        adapter=C.load_adapter(args,device).eval().requires_grad_(False)
        vae,factor=C.read_vae(args,device)
    start_epoch=0;best_score=float('inf')
    if args.resume_from:
        ck=torch.load(args.resume_from,map_location='cpu',weights_only=False)
        if name in V15 and ck.get('variant')!=name:raise ValueError('Wrong V15 variant in --resume-from')
        if name=='adapter' and 'state_dict' not in ck:raise ValueError('Expected V17 adapter checkpoint')
        if name=='bridge' and 'model' not in ck:raise ValueError('Expected V18 Bridge checkpoint')
        # Strict acquisition model and model width match.
        cfg=ck.get('config',ck)
        for key in ('mode','cycles','bg','size','seed','val_count'):
            if key in cfg and cfg[key]!=getattr(args,key):raise ValueError(f'Resume mismatch {key}')
        if cfg.get('base',args.base)!=args.base:raise ValueError('Resume mismatch base')
        model.load_state_dict(ck['model'] if name=='bridge' else ck['state_dict'],strict=True)
        if 'optimizer' in ck:
            optimizer.load_state_dict(ck['optimizer'])
            if ck.get('scaler') is not None and scaler.is_enabled():scaler.load_state_dict(ck['scaler'])
            start_epoch=int(ck['epoch']);best_score=float(ck.get('best_score',ck.get('val_score',float('inf'))))
            print(f'Exact optimizer-state resume at epoch {start_epoch}',flush=True)
        else:
            print('Warm-start from weights only; optimizer/schedule restarts at epoch 0.',flush=True)
    writer=None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(folder/'tb'/name))
    history=[];global_step=start_epoch*math.ceil(len(ds)/args.batch);bad_epochs=0
    print(f'TRAIN {name} | {device} | train {len(ds)} val {len(dv)} | epochs {args.epochs} batch {args.batch}',flush=True)
    if args.epochs<=start_epoch:raise ValueError('Total --epochs must be greater than resumed epoch.')
    try:
        for epoch in range(start_epoch,args.epochs):
            ds.epoch=epoch;model.train()
            loader=DataLoader(ds,batch_size=args.batch,shuffle=True,num_workers=args.workers,
                              pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(args.seed+epoch))
            total=0.;seen=0;start=time.perf_counter()
            for j,(gt,_) in enumerate(loader):
                if args.max_batches and j>=args.max_batches:break
                gt=gt.to(device,non_blocking=device.type=='cuda')
                budget=args.budgets[(epoch*101+j)%len(args.budgets)]
                C.seed_photons(args,j+100000*epoch,budget)
                feats,k,_,_,_=C.make_features(gt,budget,args)
                autocast=(torch.autocast('cuda',dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext())
                with autocast:loss=_training_step(model,name,gt,budget,feats,k,args,adapter,vae,factor)
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1},batch {j}')
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if scaler.is_enabled():scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
                scaler.step(optimizer);scaler.update()
                batch_size=len(gt);seen+=batch_size;total+=float(loss.detach())*batch_size
                if writer and j%args.log_every==0:writer.add_scalar('train/batch_loss',float(loss.detach()),global_step)
                global_step+=1
                if (j+1)%args.log_every==0:
                    print(f'{name} epoch={epoch+1}/{args.epochs} batch={j+1}/{len(loader)} '
                          f'loss={float(loss.detach()):.4f} speed={seen/max(time.perf_counter()-start,1e-3):.1f} im/s',flush=True)
            if seen==0:raise RuntimeError('No batches processed')
            score,vis=_training_eval(model,name,dv,args,device,adapter,vae,factor)
            ck=_write_training_checkpoint(bestfile,model,optimizer,epoch+1,args,name,score,scaler)
            improved=score<best_score-args.min_delta
            if improved:
                best_score=score;bad_epochs=0;ck['best_val_score']=best_score
                torch.save(ck,bestfile)
            else:bad_epochs+=1
            last={**ck,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict() if scaler.is_enabled() else None,
                  'best_score':best_score,'global_step':global_step}
            torch.save(last,lastfile)
            history.append({'epoch':epoch+1,'train_loss':total/seen,'val_score':score,'best_val_score':best_score})
            save_csv(folder/f'{name}_train_history.csv',history)
            if writer:
                writer.add_scalar('epoch/train_loss',total/seen,epoch+1)
                writer.add_scalar('epoch/val_score_lower_is_better',score,epoch+1)
                if (epoch+1)%args.tb_every==0:
                    for i,b,panel in vis:
                        writer.add_image(f'val/image_{i:02d}_B{b:g}_counts_BF_pred_GT',panel,epoch+1)
                    previous_epoch=ds.epoch
                    ds.epoch=0  # fixed training scenes/transforms across epochs
                    try:
                        _,train_vis=_training_eval(model,name,ds,args,device,adapter,vae,factor,max_images=args.tb_images)
                    finally:
                        ds.epoch=previous_epoch
                    for i,b,panel in train_vis:
                        writer.add_image(f'train/image_{i:02d}_B{b:g}_counts_BF_pred_GT',panel,epoch+1)
                writer.flush()
            print(f'{name}: epoch {epoch+1} train={total/seen:.5f} val={score:.5f} best={best_score:.5f}',flush=True)
            if args.patience and bad_epochs>=args.patience:
                print(f'Early stopping after {bad_epochs} epochs without improvement.',flush=True)
                break
    finally:
        if writer:writer.close()
    print('BEST:',bestfile,'LAST:',lastfile,flush=True)


def do_train(args):
    names=choose_models(args)
    valid=V15|{'adapter','bridge'}
    if any(m not in valid for m in names):
        raise ValueError(f'Trainable models: {sorted(valid)}. DDColor is frozen.')
    if args.val_images<1 or args.log_every<1:raise ValueError('val-images/log-every must be positive')
    if args.resume_from and len(names)!=1:raise ValueError('--resume-from requires one model')
    for name in names:_do_single_train(args,name)


def _load_counts(path,cycles,mode):
    """Count image HxW, or binary [T,H,W] frames; .npy/.npz only."""
    obj=np.load(path,allow_pickle=False)
    if isinstance(obj,np.lib.npyio.NpzFile):
        try:
            keys=list(obj.files)
            key='counts' if 'counts' in keys else ('frames' if 'frames' in keys else None)
            if key is None:raise ValueError(f'NPZ must contain counts or frames; found {keys}')
            data=obj[key]
        finally:obj.close()
    else:data=obj
    if data.ndim==3:
        if data.shape[0]!=cycles:raise ValueError('Binary stack length must equal --cycles')
        if not np.all((data==0)|(data==1)):raise ValueError('Binary frames must contain only 0/1')
        data=data.astype(np.float32).sum(axis=0)
    if data.ndim!=2:raise ValueError('Count input must be HxW or binary TxHxW')
    if not np.isfinite(data).all() or (data<0).any():raise ValueError('Counts must be finite and nonnegative')
    if mode=='bernoulli' and (data>cycles).any():raise ValueError('Count > cycles')
    return data.astype(np.float32)


def do_infer(args):
    if args.input is None:raise ValueError('infer requires --input PATH')
    methods=choose_models(args)
    if 'ddcolor_oracle' in methods:raise ValueError('ddcolor_oracle requires GT RGB; not valid for real inference')
    device=C.get_device(args.device)
    file=Path(args.input)
    if args.simulate:
        with Image.open(file) as im:
            im=im.convert('RGB');w,h=im.size;side=min(w,h)
            im=im.crop(((w-side)//2,(h-side)//2,(w+side)//2,(h+side)//2))
            im=im.resize((args.size,args.size),Image.Resampling.BICUBIC)
            arr=np.asarray(im,dtype=np.float32)
        gt=C.srgb_to_linear(torch.from_numpy(arr.copy()).permute(2,0,1)[None].to(device)/255.)
        C.seed_photons(args,0,args.budget,salt=999999)
        _,k,_,_,_=C.make_features(gt,args.budget,args)
        counts=k[0,0].cpu().numpy()
    else:
        counts=_load_counts(file,args.cycles,args.mode)
        if counts.shape!=(args.size,args.size):
            raise ValueError(f'Counts {counts.shape} != ({args.size},{args.size}). Do NOT resize photon counts; use matching trained --size.')
    k=torch.from_numpy(counts.copy()).to(device)[None,None]
    feats,k,est,var,bf=C.features_from_counts(k,args.budget,args)
    predictor=Predictor(args,methods)
    pred,(k,est,bf,y)=predictor.predict_features(feats,k,est,var,bf,args.budget)
    folder=args.out/'infer';folder.mkdir(parents=True,exist_ok=True)
    for m,img in pred.items():C.to_pil_srgb(img).save(folder/f'{file.stem}_{m}_B{args.budget:g}.png')
    # Counts are stored as numbers, never a gamma-transformed grayscale PNG.
    np.save(folder/f'{file.stem}_counts.npy',counts)
    imgs=[('Counts (display)',C.to_pil_srgb((k/3).clamp(0,1))),('MLE',C.to_pil_srgb(est.clamp(0,1))),
          ('MLE+BF',C.to_pil_srgb(bf.clamp(0,1)))]
    if y is not None:imgs.append(('Adapter Y',C.to_pil_srgb(y)))
    imgs.extend((m,C.to_pil_srgb(im)) for m,im in pred.items())
    if args.simulate:imgs.append(('RGB GT',C.to_pil_srgb(gt)))
    C.save_gallery([(args.budget,imgs)],args.size,folder/f'{file.stem}_gallery.png')
    print('Inference PNG and raw counts saved:',folder.resolve())


def do_plot(args):
    """Make publication-ready-ish diagnostic plots from existing CSVs, never a model inference."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:raise RuntimeError('Install matplotlib: pip install matplotlib') from e
    folder=args.out/'plots';folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    if args.plot_csv:
        paths=[Path(v) for v in args.plot_csv]
    else:
        paths=list((args.out/'rgb_'+args.split).glob('summary.csv'))
        paths+=list((args.out/'seg_voc').glob('summary.csv'))
        paths+=list(args.train_out.rglob('*_train_history.csv')) if args.train_out.exists() else []
    if not paths:raise FileNotFoundError('No summary/history CSV found; run eval/train or set --plot-csv')
    for path in paths:
        with path.open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
        if not rows:continue
        name=path.parent.name+'_'+path.stem
        if 'budget_total' in rows[0] and 'method' in rows[0]:
            key=('psnr_srgb' if 'psnr_srgb' in rows[0] else
                 'miou_gt_present' if 'miou_gt_present' in rows[0] else None)
            if key is None:continue
            fig,ax=plt.subplots(figsize=(8,4.7))
            for method in sorted(set(r['method'] for r in rows)):
                selected=sorted((r for r in rows if r['method']==method),key=lambda r:float(r['budget_total']))
                ax.plot([float(r['budget_total']) for r in selected],[float(r[key]) for r in selected],marker='o',label=method)
            ax.set(xlabel='Total signal photon budget (Y=1)',ylabel=key,title=path.parent.name)
            ax.legend(loc='best',fontsize=8);ax.grid(True,alpha=.25)
        elif 'epoch' in rows[0]:
            fig,ax=plt.subplots(figsize=(8,4.7));ax.plot([int(r['epoch']) for r in rows],
                [float(r['train_loss']) for r in rows],label='train_loss')
            ax.set(xlabel='Epoch',ylabel='Train loss',title=name)
            if 'val_score' in rows[0]:
                ax2=ax.twinx();ax2.plot([int(r['epoch']) for r in rows],
                    [float(r['val_score']) for r in rows],linestyle='--',label='val (lower better)')
                ax2.set_ylabel('Validation score')
        else:continue
        fig.tight_layout();dest=folder/(name+'.png');fig.savefig(dest,dpi=180);plt.close(fig)
        print('Figure:',dest)


LEGACY_NAMES={
    'train_v15.py','eval_v16.py','train_eval_v17.py','train_eval_v18.py','train_eval_v19.py',
    'train_eval_v20.py','eval_v18_segmentation.py','v19_seg_benchmark_fixed.py',
    'v17_v15_compat.py','v17_v16_compat.py',
    'batch_bit2bit.py','spad_mac_quickbench.py','spad_physics_v13.py',
    'spad_import.py','run_scenes.py','luma_band_diagnostic.py',
    'run_spad_pnp.py','SPAD_binomial_prox.py','SPAD_photon_holdout.py',
    'remove_files.py','rois.json',
    'RUN_SPAD_PNP.ps1','RUN_V19_BENCHMARK.ps1','RUN_V20_ALL.ps1','RUN_V20_FULL_EVAL.ps1',
}


def do_clean(args):
    root=Path(__file__).resolve().parent
    # Only archive known historical files; leave unrecognized user code untouched.
    legacy=[p for p in root.iterdir() if p.is_file() and p.name in LEGACY_NAMES]
    trash=[p for p in root.iterdir() if p.is_file() and (p.name.startswith('._') or p.name=='.DS_Store')]
    print('Historical files to archive/remove from project root:')
    for p in legacy:print('  archive:',p.name)
    for p in trash:print('  trash:  ',p.name)
    if not args.apply:
        print('DRY RUN only. Execute: python spad.py clean --apply')
        return
    # Verify key original model checkpoints before hiding original source files.
    do_check(args)
    archive=root/'archive';archive.mkdir(exist_ok=True)
    dest=archive/('legacy_source_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in legacy:z.write(p,arcname=p.name)
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:raise RuntimeError('Archive CRC verification failed')
        if set(z.namelist())!={p.name for p in legacy}:raise RuntimeError('Archive file list mismatch')
    # Never touch checkpoint/data/external/papers directories.
    for p in legacy + trash:p.unlink()
    print('CLEANUP COMPLETE. Previous source code recoverable from:',dest)
    print('Kept: spad.py, spad_core.py, runs/, data/, external/, and all unknown user files.')


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('action',choices=['check','train','infer','rgb','seg','plot','clean'])
    p.add_argument('--models',nargs='+',choices=METHODS+('adapter',),default=None)
    p.add_argument('--model',dest='one_model',action='append',choices=METHODS+('adapter',))
    p.add_argument('--data',type=Path,default=Path('data/imagenette2-160'))
    p.add_argument('--voc-root',type=Path,default=Path('data/VOC'))
    p.add_argument('--v15-out',type=Path,default=Path('runs/v15_64'))
    p.add_argument('--v17-out',type=Path,default=Path('runs/v17_adapter'))
    p.add_argument('--v18-out',type=Path,default=Path('runs/v18_bridge_fast'))
    p.add_argument('--out',type=Path,default=Path('runs/spad_final'))
    p.add_argument('--mode',choices=['bernoulli','poisson'],default='bernoulli')
    p.add_argument('--cycles',type=int,default=64);p.add_argument('--bg',type=float,default=.01)
    p.add_argument('--budgets',nargs='+',type=float,default=[.5,1,2,4,8])
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--val-count',type=int,default=500)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--bf-radius',type=int,default=3);p.add_argument('--bf-sigma',type=float,default=1.5)
    p.add_argument('--bf-knoise',type=float,default=2.);p.add_argument('--bf-prec-cap',type=float,default=4.)
    p.add_argument('--split',choices=['test','val'],default='test')
    p.add_argument('--eval-images',type=int,default=20,help='0=all images')
    p.add_argument('--skip-images',type=int,default=6,help='VOC val images previously used for checkpoint selection')
    p.add_argument('--gallery-n',type=int,default=6);p.add_argument('--gallery-tile',type=int,default=160)
    p.add_argument('--segmentor',choices=['resnet50','mobilenet'],default='resnet50')
    p.add_argument('--ddcolor-repo',type=Path,default=Path('external/DDColor'))
    p.add_argument('--ddcolor-model',default='piddnad/ddcolor_paper_tiny')
    p.add_argument('--ddcolor-weights',type=Path,default=None)
    p.add_argument('--ddcolor-model-size',choices=['tiny','large'],default='tiny')
    p.add_argument('--ddcolor-size',type=int,default=256)
    p.add_argument('--vae-model',default='stabilityai/sd-vae-ft-mse')
    p.add_argument('--steps',type=int,default=4);p.add_argument('--luma-fusion',type=float,default=.5)
    p.add_argument('--train-out',type=Path,default=Path('runs/spad_training'),help='Never overwrites original runs by default')
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--base',type=int,default=32,help='V15/V17: 32, V18: 64')
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--val-images',type=int,default=24);p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--lambda-y',type=float,default=.5)
    p.add_argument('--lambda-edge',type=float,default=.25)
    p.add_argument('--lambda-ph',type=float,default=.02)
    p.add_argument('--lambda-rgb',type=float,default=.1)
    p.add_argument('--noise-sigma',type=float,default=.05)
    p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--min-delta',type=float,default=1e-4)
    p.add_argument('--amp',choices=['none','fp16','bf16'],default='none')
    p.add_argument('--resume-from',type=Path,default=None)
    p.add_argument('--overwrite',action='store_true',help='Allow overwrite in --train-out only')
    p.add_argument('--tensorboard',action='store_true');p.add_argument('--tb-every',type=int,default=1)
    p.add_argument('--tb-images',type=int,default=2);p.add_argument('--log-every',type=int,default=25)
    p.add_argument('--input',type=Path,help='For infer: counts .npy/.npz or RGB image with --simulate')
    p.add_argument('--simulate',action='store_true',help='For infer: simulate SPAD from an RGB image')
    p.add_argument('--budget',type=float,default=8.,help='For infer: calibrated total photon budget')
    p.add_argument('--plot-csv',nargs='+',default=None,help='CSV paths for plot')
    p.add_argument('--apply',action='store_true',help='Only for clean; ZIP archive then remove known old root source files')
    return p


def main():
    args=parser().parse_args()
    if args.size<32 or args.size%32:raise ValueError('--size must be a multiple of 32')
    if args.cycles<1 or any(b<=0 for b in args.budgets):raise ValueError('Invalid cycles or budget')
    if args.eval_images<0 or args.gallery_n<0:raise ValueError('eval-images/gallery-n must be >=0')
    if args.budget<=0 or args.epochs<=0 or args.batch<=0 or args.base<8 or args.base%8!=0:
        raise ValueError('budget/epochs/batch must be positive; base must be divisible by 8')
    {'check':do_check,'train':do_train,'infer':do_infer,'rgb':do_rgb,'seg':do_seg,'plot':do_plot,'clean':do_clean}[args.action](args)

if __name__=='__main__':main()
