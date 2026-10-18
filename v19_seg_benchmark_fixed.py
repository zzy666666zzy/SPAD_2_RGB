#!/usr/bin/env python3
"""V19 segmentation benchmark v2: GT sanity check and paired RGB recon comparisons.

Two modes:
  sanity  -- VOC original RGB vs matched center-crop RGB, with official torchvision
             pretrained DeepLab weights.transforms() and an optional legacy 256 input.
  compare -- same crop and EXACT SAME photon measurements for V15/V18/V19,
             using official pretrained segmentation preprocessing for EVERY method.

Does NOT train or modify any checkpoints; does NOT use VOC val for fitting.
Requires existing train_eval_v19.py, train_eval_v18.py and compatibility modules.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

import train_eval_v19 as v19
import train_eval_v18 as v18

METHODS = ('raw','mle','full','ddcolor_mle_bf','ddcolor_oracle','ddcolor_full_luma',
           'ddcolor_adapter','bridge','bridge_fused','vae_gray','vae_oracle',
           'v19_a','v19_b','v19_a_fused','v19_b_fused')
VOC_CLASSES = ('background','aeroplane','bicycle','bird','boat','bottle','bus','car',
               'cat','chair','cow','diningtable','dog','horse','motorbike','person',
               'pottedplant','sheep','sofa','train','tvmonitor')

def voc_colors():
    colors=np.zeros((256,3),dtype=np.uint8)
    for label in range(21):
        v=label
        for bit in range(8):
            for channel in range(3):
                colors[label,channel]|=((v>>channel)&1)<<(7-bit)
            v>>=3
    colors[255]=[255,255,255]
    return colors

PALETTE=voc_colors()

def mask_pil(a):
    a=np.asarray(a,dtype=np.int64)
    valid=(a>=0)&(a<PALETTE.shape[0]);tmp=np.full(a.shape,255,np.uint8)
    tmp[valid]=a[valid].astype(np.uint8)
    return Image.fromarray(PALETTE[tmp],mode='RGB')

def pil_of_srgb(rgb):
    if rgb.ndim==4:rgb=rgb[0]
    a=(rgb.detach().float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    return Image.fromarray(a,'RGB')

def pil_of_linear(rgb):
    return pil_of_srgb(v18.v15.linear_to_srgb(rgb))

def palette_mosaic(path, examples, column_names, tile=160, title='VOC comparison'):
    if not examples:return
    # each row: (label, [(PIL rgb/mask, interpolation-nearest-bool), ...])
    header=55;gap=8;pad=12;row_header=26
    width=pad*2+len(column_names)*(tile+gap)
    h=header+len(examples)*(tile+row_header+gap)+pad
    out=Image.new('RGB',(width,h),'white');d=ImageDraw.Draw(out)
    try:font=ImageFont.truetype('DejaVuSans.ttf',13)
    except OSError:font=ImageFont.load_default()
    d.text((pad,3),title[:125],fill='black',font=font)
    for col,name in enumerate(column_names):d.text((pad+col*(tile+gap),30),name[:23],fill='black',font=font)
    for i,(label,tiles) in enumerate(examples):
        if len(tiles)!=len(column_names):raise ValueError('Column mismatch in gallery')
        y=header+i*(tile+row_header+gap)
        d.text((pad,y),label,fill='black',font=font);y+=row_header
        for j,(im,is_mask) in enumerate(tiles):
            resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
            out.paste(im.resize((tile,tile),resample),(pad+j*(tile+gap),y))
    path.parent.mkdir(parents=True,exist_ok=True);out.save(path)

def update_cm(cm,pred,truth):
    if pred.shape!=truth.shape:raise ValueError(f'Prediction/mask size mismatch: {pred.shape}/{truth.shape}')
    mask=(truth!=255)&(truth>=0)&(truth<21)&(pred>=0)&(pred<21)
    bins=np.bincount(truth[mask].astype(np.int64)*21+pred[mask].astype(np.int64),minlength=441)
    cm+=bins.reshape(21,21)

def fixed_metrics(cm,classes=None):
    """Uses same set of *GT-present* classes for all methods; no denominator changes from extra FP labels."""
    tp=np.diag(cm).astype(float)
    union=cm.sum(0)+cm.sum(1)-tp
    if classes is None:classes=(cm.sum(1)>0)
    classes=np.asarray(classes,bool)
    good=classes & (union>0)
    fg=good.copy();fg[0]=False
    iou=tp/np.maximum(union,1)
    return dict(miou_gt_present=float(np.mean(iou[good])) if good.any() else float('nan'),
                foreground_miou_gt_present=float(np.mean(iou[fg])) if fg.any() else float('nan'),
                pixel_accuracy=float(tp.sum()/max(float(cm.sum()),1)),
                gt_present_classes=int(good.sum()),evaluated_pixels=int(cm.sum()))

def tensor_pil(im,device):
    a=np.asarray(im.convert('RGB'),dtype=np.float32).copy()
    return torch.from_numpy(a).permute(2,0,1)[None].to(device)/255.

@torch.inference_mode()
def official_predict(model,transform,srgb,mask_size):
    # TorchVision DeepLabV3_Weights.transforms(): aspect-preserving resize to 520,
    # ImageNet normalization. Always resize logits back to the ORIGINAL mask shape.
    x=transform(srgb[0].float().clamp(0,1)).unsqueeze(0)
    logits=model(x)['out']
    return F.interpolate(logits.float(),size=mask_size,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)

@torch.inference_mode()
def legacy_predict(model,srgb,mask_size,side=256):
    # Diagnostic for old V19 evaluation, not the official inference transform.
    x=F.interpolate(srgb.float().clamp(0,1),size=(side,side),mode='bilinear',align_corners=False)
    mu=x.new_tensor([.485,.456,.406])[None,:,None,None]
    sd=x.new_tensor([.229,.224,.225])[None,:,None,None]
    logits=model((x-mu)/sd)['out']
    return F.interpolate(logits.float(),size=mask_size,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)

def load_voc(root):
    from torchvision.datasets import VOCSegmentation
    return VOCSegmentation(root=str(root),year='2012',image_set='val',download=False)

def get_crop(img,mask,size):
    # EXACT same center crop used to create the V19 input and corresponding GT mask.
    return v19.segutil.crop_pair(img.convert('RGB'),mask,size)

def get_segmentor(device,name):
    # weights and recommended official transforms come from the SAME enum.
    return v19.segutil.get_segmentor(device,name)

def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def get_indices(ds,args):
    start=args.skip_images
    if start>=len(ds):raise ValueError(f'--skip-images {start} exceeds VOC val length {len(ds)}')
    stop=len(ds) if args.eval_images==0 else min(len(ds),start+args.eval_images)
    return range(start,stop)

@torch.inference_mode()
def run_sanity(args):
    device=v18.device_for(args)
    ds=load_voc(args.voc_root)
    segmentor,official=get_segmentor(device,args.segmentor)
    names=['GT_original_official','GT_crop_official','GT_crop_legacy256']
    cms={m:np.zeros((21,21),np.int64) for m in names}
    rows=[];gallery=[];indices=get_indices(ds,args);n=len(indices)
    print(f'Using VOC val indices {indices.start}..{indices.stop-1} (excluding checkpoint-selection images)',flush=True)
    for done,idx in enumerate(indices,1):
        image,labels=ds[idx]
        crop,mask_crop=get_crop(image,labels,args.size)
        label_orig=np.asarray(labels,dtype=np.uint8)
        label_crop=np.asarray(mask_crop,dtype=np.uint8)
        x_orig=tensor_pil(image,device);x_crop=tensor_pil(crop,device)
        ps={'GT_original_official':official_predict(segmentor,official,x_orig,label_orig.shape),
            'GT_crop_official':official_predict(segmentor,official,x_crop,label_crop.shape),
            'GT_crop_legacy256':legacy_predict(segmentor,x_crop,label_crop.shape)}
        targets={names[0]:label_orig,names[1]:label_crop,names[2]:label_crop}
        for m in names:
            local=np.zeros((21,21),np.int64);update_cm(local,ps[m],targets[m]);cms[m]+=local
            rows.append(dict(image_index=idx,method=m,**fixed_metrics(local,classes=np.bincount(targets[m][targets[m]<21],minlength=21)>0)))
        if len(gallery)<args.gallery_n:
            gallery.extend([
                (f'VOC {idx} | input RGB',[(image,False),(crop,False),(crop,False),(crop,False)]),
                (f'VOC {idx} | GT and prediction masks',[(mask_pil(label_orig),True),(mask_pil(label_crop),True),
                                    (mask_pil(ps['GT_crop_official']),True),(mask_pil(ps['GT_crop_legacy256']),True)]),
            ])
        if done%10==0 or done==n:print(f'GT sanity {done}/{n}',flush=True)
    gt_names={m:cms[m].sum(1)>0 for m in names}
    summary=[dict(method=m,images=n,**fixed_metrics(cms[m],gt_names[m])) for m in names]
    write_rows(args.out/'sanity_summary.csv',summary)
    write_rows(args.out/'sanity_per_image.csv',rows)
    palette_mosaic(args.out/'sanity_gallery.png',gallery,['Native VOC','128 crop GT','Official DeepLab','Legacy DeepLab'],args.tile,'GT RGB / mask sanity; native vs 128 crop vs legacy preprocessing')
    np.savez_compressed(args.out/'sanity_confusions.npz',**cms)
    print('\n=== GT SANITY SUMMARY ===')
    for r in summary:print(f"{r['method']:23s} mIoU={r['miou_gt_present']:.3f} foreground={r['foreground_miou_gt_present']:.3f} pixel_acc={r['pixel_accuracy']:.3f}")
    print('Native / crop use different spatial content; compare qualitative change, not paired PSNR.')
    print('Saved to',args.out,flush=True)

@torch.inference_mode()
def run_compare(args):
    device=v18.device_for(args)
    ds=load_voc(args.voc_root)
    segmentor,official=get_segmentor(device,args.segmentor)
    # V19 existing compatible model loader, using exact same V15/V18/V19 weights.
    cfg=argparse.Namespace(**vars(args))
    cfg.out=args.v19_out
    cfg.v18_out=args.v18_out
    cfg.v15_out=args.v15_out
    cfg.v17_out=args.v17_out
    cfg.methods=args.methods
    models=v19.load_for_eval(cfg,device)
    methods=['GT_crop_official']+args.methods
    cms={(m,b):np.zeros((21,21),np.int64) for b in args.budgets for m in methods}
    native_cm=np.zeros((21,21),np.int64)
    rows=[];indices=get_indices(ds,args);n=len(indices); galleries={b:[] for b in args.budgets}
    print(f'Using VOC val indices {indices.start}..{indices.stop-1} (excluding checkpoint-selection images)',flush=True)
    for done,idx in enumerate(indices,1):
        img,lab=ds[idx]
        crop,msk=get_crop(img,lab,args.size)
        truth=np.asarray(msk,dtype=np.uint8)
        srgb=tensor_pil(crop,device)
        linear=v18.v15.srgb_to_linear(srgb)
        gtpred=official_predict(segmentor,official,srgb,truth.shape)
        if args.native_reference:
            native_gt=np.asarray(lab,dtype=np.uint8)
            native=official_predict(segmentor,official,tensor_pil(img,device),native_gt.shape)
            update_cm(native_cm,native,native_gt)
        for budget in args.budgets:
            outputs,_=v19.all_outputs(linear,idx,budget,cfg,device,models)
            pred_masks={'GT_crop_official':gtpred}
            for method in args.methods:
                display=v18.v15.linear_to_srgb(outputs[method])
                pred_masks[method]=official_predict(segmentor,official,display,truth.shape)
            for name,pred in pred_masks.items():
                one=np.zeros((21,21),np.int64);update_cm(one,pred,truth)
                cms[(name,budget)]+=one
                present=np.bincount(truth[truth<21],minlength=21)>0
                rows.append(dict(image_index=idx,budget_total=budget,method=name,**fixed_metrics(one,present)))
            if len(galleries[budget])<args.gallery_n:
                # TWO rows per sample: original RGB, reconstructed RGB; GT / pred masks.
                # Each method column uses the exact same photon realization.
                label=f'VOC {idx:03d} | B={budget:g}'
                ordered=['GT_crop_official']+args.methods
                galleries[budget].append((label+' | RGB',[(crop,False),(crop,False)]+[(pil_of_linear(outputs[m]),False) for m in args.methods]))
                galleries[budget].append((label+' | Segmentation',[(mask_pil(truth),True),(mask_pil(gtpred),True)]+[(mask_pil(pred_masks[m]),True) for m in args.methods]))
        if done%5==0 or done==n:print(f'Paired comparison {done}/{n}',flush=True)
    summary=[]
    gt_class_set=(cms[('GT_crop_official',args.budgets[0])].sum(1)>0)
    for b in args.budgets:
        for m in methods:
            summary.append(dict(budget_total=b,method=m,images=n,
                                **fixed_metrics(cms[(m,b)],gt_class_set)))
        if galleries[b]:
            palette_mosaic(args.out/f'seg_gallery_B{b:g}.png',galleries[b],
                           ['VOC GT','DeepLab GT']+args.methods,args.tile,
                           f'VOC | Photon budget B={b:g} | RGB top / Seg below | OFFICIAL DeepLab')
    write_rows(args.out/'compare_summary.csv',summary)
    write_rows(args.out/'compare_per_image.csv',rows)
    np.savez_compressed(args.out/'compare_confusions.npz',**{f'{m}_B{b:g}':cm for (m,b),cm in cms.items()})
    if args.native_reference:
        write_rows(args.out/'native_gt_reference.csv',[
            dict(method='GT_original_official',images=n,**fixed_metrics(native_cm,native_cm.sum(1)>0))])
    (args.out/'config.json').write_text(json.dumps({**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             'official_preprocessing':'TorchVision DeepLab weights.transforms(), logits aligned to VOC mask',
             'metric':'mIoU using fixed set of dataset GT-present classes (same across methods)',
             'gt_luminance_oracles':'Do not compare oracle methods as deployable reconstructions'},indent=2),encoding='utf8')
    print('\n=== PAIRED VOC SEGMENTATION COMPARISON ===')
    for r in summary:
        print(f"B={r['budget_total']:g} {r['method']:18s} foreground mIoU={r['foreground_miou_gt_present']:.3f} acc={r['pixel_accuracy']:.3f}")
    print('Saved to',args.out,flush=True)

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('action',choices=['sanity','compare'])
    p.add_argument('--voc-root',type=Path,default=Path('data/VOC'))
    p.add_argument('--v19-out',type=Path,default=Path('runs/v19_seg'))
    p.add_argument('--v18-out',type=Path,default=Path('runs/v18_bridge_fast'))
    p.add_argument('--v15-out',type=Path,default=Path('runs/v15_64'))
    p.add_argument('--v17-out',type=Path,default=Path('runs/v17_adapter'))
    p.add_argument('--out',type=Path,default=Path('runs/v19_seg/benchmark_fixed'))
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    p.add_argument('--segmentor',choices=['deeplab_mobilenet','deeplab_resnet50'],default='deeplab_resnet50')
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--seg-input-size',type=int,default=520,help='Only inherited V19 compatibility arg; official transforms determine actual network input')
    p.add_argument('--budgets',type=float,nargs='+',default=[.5,2,8])
    p.add_argument('--mode',choices=['bernoulli','poisson'],default='bernoulli')
    p.add_argument('--cycles',type=int,default=64)
    p.add_argument('--bg',type=float,default=.01)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--val-count',type=int,default=500)
    p.add_argument('--vae-model',default='stabilityai/sd-vae-ft-mse')
    p.add_argument('--base',type=int,default=64)
    p.add_argument('--sample-noise',type=float,default=0.)
    p.add_argument('--bf-radius',type=int,default=3)
    p.add_argument('--bf-sigma',type=float,default=1.5)
    p.add_argument('--bf-knoise',type=float,default=2.)
    p.add_argument('--bf-prec-cap',type=float,default=4.)
    p.add_argument('--ddcolor-repo',type=Path,default=Path('external/DDColor'))
    p.add_argument('--ddcolor-model',default='piddnad/ddcolor_paper_tiny')
    p.add_argument('--ddcolor-weights',type=Path,default=None)
    p.add_argument('--ddcolor-model-size',choices=['tiny','large'],default='tiny')
    p.add_argument('--ddcolor-size',type=int,default=256)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--luma-fusion',type=float,default=.5)
    p.add_argument('--checkpoint',choices=['best','last'],default='best')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=['raw','full','bridge','v19_a','v19_b'])
    p.add_argument('--eval-images',type=int,default=20,help='0=all remaining VOC val images after --skip-images')
    p.add_argument('--skip-images',type=int,default=0,help='Skip checkpoint-selection VOC val indices; V19 used --val-images 6 in supplied training commands')
    p.add_argument('--gallery-n',type=int,default=6)
    p.add_argument('--tile',type=int,default=160)
    p.add_argument('--native-reference',action='store_true',help='Compare original uncropped VOC GT independently')
    args=p.parse_args(argv)
    if args.eval_images<0 or args.gallery_n<0 or args.skip_images<0 or args.size%32:p.error('Invalid image counts or size')
    if any(x<=0 for x in args.budgets):p.error('Budgets must be positive')
    return args

def main():
    args=parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.action=='sanity':run_sanity(args)
    else:run_compare(args)

if __name__=='__main__':main()
