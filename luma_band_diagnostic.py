#!/usr/bin/env python3
"""Zero-training diagnostic: which spatial band of photon-estimated luminance limits DDColor and VOC segmentation?

This is a non-deployable diagnostic whenever GT luminance is mixed in.
Reuses V17 Adapter, pretrained DDColor, and benchmark's official DeepLab preprocessing.
No checkpoints are altered.

Run from the existing low_count_to_rgb repo, alongside v19_seg_benchmark_fixed.py.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import v19_seg_benchmark_fixed as bench
import train_eval_v18 as v18


def gaussian_lowpass(y, sigma=2.0):
    if sigma <= 0:
        raise ValueError('sigma must be > 0')
    radius = int(math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=y.device, dtype=y.dtype)
    w = torch.exp(-0.5 * (x / sigma) ** 2)
    w = w / w.sum()
    kernel = (w[:, None] * w[None, :]).view(1, 1, len(w), len(w))
    # Use replicated borders for stability at small sizes.
    return F.conv2d(F.pad(y, [radius] * 4, mode='replicate'), kernel)


def variants(estimated, ground_truth, sigma=2.0, sharpen=0.5):
    """All inputs and outputs are LINEAR-light Y tensors, same spatial resolution.

    GT_high: estimated low-frequency + GT high-frequency, oracle only.
    GT_low: GT low-frequency + estimated high-frequency, oracle only.
    Unsharp: no GT leakage, deployable post-processing baseline.
    """
    e_low = gaussian_lowpass(estimated, sigma)
    g_low = gaussian_lowpass(ground_truth, sigma)
    return {
        'adapter': estimated.clamp(0, 1),
        'unsharp': (estimated + sharpen * (estimated - e_low)).clamp(0, 1),
        'GT_low_oracle': (g_low + estimated - e_low).clamp(0, 1),
        'GT_high_oracle': (e_low + ground_truth - g_low).clamp(0, 1),
        'GT_Y_oracle': ground_truth.clamp(0, 1),
    }


def edge_error(a, b):
    dx = (a[..., :, 1:] - a[..., :, :-1] - b[..., :, 1:] + b[..., :, :-1]).abs().mean()
    dy = (a[..., 1:, :] - a[..., :-1, :] - b[..., 1:, :] + b[..., :-1, :]).abs().mean()
    return float(0.5 * (dx + dy))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--voc-root', type=Path, default=Path('data/VOC'))
    p.add_argument('--v17-out', type=Path, default=Path('runs/v17_adapter'))
    p.add_argument('--ddcolor-repo', type=Path, default=Path('external/DDColor'))
    p.add_argument('--out', type=Path, default=Path('runs/luma_band_diagnostic'))
    p.add_argument('--device', choices=['cuda','cpu','auto'], default='cuda')
    p.add_argument('--segmentor', choices=['deeplab_resnet50','deeplab_mobilenet'], default='deeplab_resnet50')
    p.add_argument('--mode', choices=['bernoulli','poisson'], default='bernoulli')
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--cycles', type=int, default=64)
    p.add_argument('--bg', type=float, default=0.01)
    p.add_argument('--budget', type=float, default=8.0)
    p.add_argument('--sigma', type=float, default=2.0)
    p.add_argument('--sharpen', type=float, default=0.5)
    p.add_argument('--skip-images', type=int, default=6)
    p.add_argument('--eval-images', type=int, default=20, help='0 means all remaining VOC validation images')
    p.add_argument('--gallery-n', type=int, default=5)
    p.add_argument('--tile', type=int, default=160)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-count', type=int, default=500)
    p.add_argument('--bf-radius', type=int, default=3)
    p.add_argument('--bf-sigma', type=float, default=1.5)
    p.add_argument('--bf-knoise', type=float, default=2.)
    p.add_argument('--bf-prec-cap', type=float, default=4.)
    p.add_argument('--ddcolor-model', default='piddnad/ddcolor_paper_tiny')
    p.add_argument('--ddcolor-weights', type=Path)
    p.add_argument('--ddcolor-model-size', choices=['tiny','large'], default='tiny')
    p.add_argument('--ddcolor-size', type=int, default=256)
    args = p.parse_args()
    if args.size % 32 or args.skip_images < 6 or args.budget <= 0 or args.eval_images < 0:
        p.error('--size must be multiple of 32; --skip-images >= 6; budget positive; eval-images >=0')
    args.out.mkdir(parents=True, exist_ok=True)
    device = v18.device_for(args)
    voc = bench.load_voc(args.voc_root)
    seg, seg_transform = bench.get_segmentor(device, args.segmentor)
    adapter = v18.adapter_load(args, device)
    colorizer, _ = v18.v16.load_ddcolor(args, device)
    end = len(voc) if args.eval_images == 0 else min(len(voc), args.skip_images + args.eval_images)
    if end <= args.skip_images:
        raise ValueError('No VOC validation images in chosen range')
    order = ['adapter','unsharp','GT_low_oracle','GT_high_oracle','GT_Y_oracle']
    names = ['GT RGB'] + order
    cms = {x: np.zeros((21,21), np.int64) for x in ['GT_RGB'] + order}
    lum_rows = []
    gallery = []
    budget = args.budget
    for idx in range(args.skip_images,end):
        image, label = voc[idx]
        crop, mask = bench.get_crop(image,label,args.size)
        ymask = np.asarray(mask,dtype=np.uint8)
        srgb = bench.tensor_pil(crop,device)
        linear = v18.v15.srgb_to_linear(srgb)
        ytrue = v18.v15.luminance(linear)
        v18.v15.seed_photons(args,idx,budget,salt=300000)
        feats,_,_,_,_ = v18.v15.make_features(linear,budget,args)
        yhat = adapter(feats).clamp(0,1)
        preds = variants(yhat,ytrue,args.sigma,args.sharpen)
        gt_seg = bench.official_predict(seg,seg_transform,srgb,ymask.shape)
        bench.update_cm(cms['GT_RGB'],gt_seg,ymask)
        for m in order:
            yy = preds[m]
            recovered = v18.v16.ddcolorize(colorizer,yy,device)
            recovered_srgb = v18.v15.linear_to_srgb(recovered)
            pred_mask = bench.official_predict(seg,seg_transform,recovered_srgb,ymask.shape)
            bench.update_cm(cms[m],pred_mask,ymask)
            lum_rows.append({'image_index':idx,'method':m,'budget_total':budget,
                             'linear_Y_MSE':float(F.mse_loss(yy,ytrue)),
                             'linear_Y_edge_MAE':edge_error(yy,ytrue)})
            if len(gallery) < args.gallery_n*3:
                # Store each output for this index without needing a second inference.
                preds[m] = (yy, recovered, pred_mask)
        if len(gallery) < args.gallery_n*3:
            gallery.append((f'VOC {idx:03d} | linear Y (shown as sRGB)',
                             [(bench.pil_of_srgb(v18.v15.linear_to_srgb(ytrue.repeat(1,3,1,1))),False)]+
                             [(bench.pil_of_srgb(v18.v15.linear_to_srgb(preds[m][0].repeat(1,3,1,1))),False) for m in order]))
            gallery.append((f'VOC {idx:03d} | RGB',[(crop,False)]+[(bench.pil_of_linear(preds[m][1]),False) for m in order]))
            gallery.append((f'VOC {idx:03d} | segmentation',[(bench.mask_pil(ymask),True)]+[(bench.mask_pil(preds[m][2]),True) for m in order]))
        if (idx-args.skip_images+1)%5==0 or idx+1==end:
            print(f'Diagnostic: {idx-args.skip_images+1}/{end-args.skip_images}',flush=True)
    present = cms['GT_RGB'].sum(axis=1)>0
    rows=[]
    for m in ['GT_RGB'] + order:
        ys=[r for r in lum_rows if r['method']==m]
        rows.append({'method':m, 'images':end-args.skip_images,
                     **bench.fixed_metrics(cms[m],present),
                     'mean_linear_Y_MSE':float(np.mean([r['linear_Y_MSE'] for r in ys])) if ys else 0.,
                     'mean_linear_Y_edge_MAE':float(np.mean([r['linear_Y_edge_MAE'] for r in ys])) if ys else 0.})
    bench.write_rows(args.out/'summary.csv',rows)
    bench.write_rows(args.out/'luminance_per_image.csv',lum_rows)
    bench.palette_mosaic(args.out/'gallery.png', gallery, names, tile=args.tile,
                         title=f'Photon Y spatial-band ablation | B={budget:g} | sigma={args.sigma:g} | ORACLE variants use GT luminance')
    np.savez_compressed(args.out/'confusions.npz',**cms)
    print('\n=== FIXED GT-PRESENT-CLASS MIOU ===')
    for r in rows:
        print(f"{r['method']:18s} mIoU={r['miou_gt_present']:.4f} FG={r['foreground_miou_gt_present']:.4f} Y_MSE={r['mean_linear_Y_MSE']:.5f}")
    print('All GT_low/GT_high/GT_Y branches are diagnostic oracles; do not report as deployable.',flush=True)
    print('Saved:',args.out,flush=True)


if __name__=='__main__':
    main()
