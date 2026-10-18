#!/usr/bin/env python3
"""V16: reproducible static photon-to-RGB benchmark retaining V15 baselines.

Evaluate V15 raw/mle/full checkpoint outputs *and* a pretrained DDColor prior:
- ddcolor_oracle: exact GT linear luminance -> pretrained DDColor (diagnostic only)
- ddcolor_mle_bf: V15 MLE + bilateral grayscale -> pretrained DDColor
- ddcolor_full_luma: V15 full network predicted luminance -> pretrained DDColor

No training here. No GT pixels are supplied to non-oracle inference methods.
This is not a latent bridge. A pretrained colour-prior control should precede one.

Install official DDColor repo via git clone -- see V16_README.md.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import torch

import v17_v15_compat as v15

ALL_METHODS = (
    'raw', 'mle', 'full',
    'ddcolor_mle_bf', 'ddcolor_oracle', 'ddcolor_full_luma'
)
DDCOLOR_METHODS = set(ALL_METHODS) - {'raw', 'mle', 'full'}
LEGACY_METHODS = {'raw', 'mle', 'full'}


def load_ddcolor(args, device):
    """Use the *official* DDColor implementation and colourization pipeline.

    HF downloads real pretrained weights the first time and caches them.
    Local weights supported by the official build_ddcolor_model function.
    """
    repo = args.ddcolor_repo.resolve()
    if not (repo / 'ddcolor' / '__init__.py').is_file():
        raise FileNotFoundError(
            f'DDColor source not found at {repo}. Run: '
            'git clone https://github.com/piddnad/DDColor.git external/DDColor'
        )
    sys.path.insert(0, str(repo))
    try:
        from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model
    except Exception as e:
        raise RuntimeError(
            'Cannot import the official DDColor source. Install cv2 and DDColor '
            'requirements in your current PyTorch environment; see V16_README.md'
        ) from e
    if args.ddcolor_weights:
        model = build_ddcolor_model(
            DDColor, model_path=str(args.ddcolor_weights.resolve()),
            input_size=args.ddcolor_size,
            model_size=args.ddcolor_model_size, device=device,
        )
        checkpoint_name = str(args.ddcolor_weights.resolve())
    else:
        try:
            from huggingface_hub import PyTorchModelHubMixin
        except ImportError as e:
            raise RuntimeError('Install huggingface-hub: python -m pip install huggingface-hub') from e

        class DDColorHF(DDColor, PyTorchModelHubMixin):
            def __init__(self, config=None, **kwargs):
                if isinstance(config, dict):
                    kwargs = {**config, **kwargs}
                super().__init__(**kwargs)

        name = args.ddcolor_model
        if '/' not in name and not Path(name).is_dir():
            name = 'piddnad/' + name
        print(f'Loading pretrained DDColor {name} (first run may download weights)', flush=True)
        model = DDColorHF.from_pretrained(name).to(device).eval()
        checkpoint_name = name
    model.eval()
    return ColorizationPipeline(model, input_size=args.ddcolor_size, device=device), checkpoint_name


def luminance_to_uint8_bgr(y_linear):
    """Convert a *linear-light luminance* tensor (1,1,H,W) into neutral sRGB BGR.

    This is important: blindly converting linear Y to uint8 would make the DDColor
    brightness distribution inconsistent with how it was pretrained.
    """
    import cv2
    if y_linear.ndim != 4 or y_linear.shape[0] != 1 or y_linear.shape[1] != 1:
        raise ValueError('Expected linear luminance tensor shape (1,1,H,W)')
    gray_srgb = v15.linear_to_srgb(y_linear.clamp(0, 1))[0, 0].cpu().numpy()
    rgb8 = np.repeat(np.rint(gray_srgb[..., None] * 255).astype(np.uint8), 3, axis=-1)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)


def ddcolorize(colorizer, y_linear, device):
    """Return RGB prediction as *linear* PyTorch (1,3,H,W), not BGR."""
    import cv2
    img_bgr = luminance_to_uint8_bgr(y_linear)
    out_bgr = colorizer.process(img_bgr)
    if out_bgr.shape != img_bgr.shape:
        raise ValueError(f'DDColor returned shape {out_bgr.shape}, expected {img_bgr.shape}')
    rgb_srgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    p = torch.from_numpy(rgb_srgb.copy()).permute(2, 0, 1)[None].to(device)
    return v15.srgb_to_linear(p).clamp(0, 1)


def load_v15(args, device, variant):
    pth = args.v15_out / f'{variant}_best.pt'
    if not pth.is_file():
        raise FileNotFoundError(f'Missing V15 checkpoint: {pth}. Use V15 training first.')
    try:
        ck = torch.load(pth, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(pth, map_location=device)
    for key in ('mode','cycles','bg','size','seed','val_count',
                'bf_radius','bf_sigma','bf_knoise','bf_prec_cap'):
        if key not in ck:
            raise KeyError(f'V15 checkpoint missing configuration {key}: {pth}')
        if ck[key] != getattr(args, key):
            raise ValueError(
                f'{variant}: checkpoint {key}={ck[key]} != requested '
                f'{getattr(args,key)}. Use EXACT original V15 acquisition options.'
            )
    net = v15.RGBUNet(ck['base']).to(device)
    net.load_state_dict(ck['state_dict'], strict=True)
    net.eval()
    print(f'Loaded {variant} checkpoint epoch={ck["epoch"]}: {pth}', flush=True)
    return net


def make_variant(full, variant):
    x = full.clone()
    if variant == 'raw':
        x[:,1:4] = 0
    elif variant == 'mle':
        x[:,0:1] = 0
        x[:,3:4] = 0
    elif variant != 'full':
        raise ValueError(variant)
    return x


def ssim_srgb(a, b):
    """Gaussian-window RGB SSIM, per-channel mean, all inputs sRGB [0,1]."""
    import cv2
    C1, C2 = (0.01)**2, (0.03)**2
    total = 0.0
    for c in range(3):
        x, y = a[...,c], b[...,c]
        smooth = lambda z: cv2.GaussianBlur(z, (11,11), sigmaX=1.5,
                                             borderType=cv2.BORDER_REFLECT_101)
        mx, my = smooth(x), smooth(y)
        vx = np.maximum(smooth(x*x)-mx*mx, 0)
        vy = np.maximum(smooth(y*y)-my*my, 0)
        xy = smooth(x*y)-mx*my
        s = ((2*mx*my+C1)*(2*xy+C2))/((mx*mx+my*my+C1)*(vx+vy+C2))
        total += float(s.mean())
    return total/3


def compute_metrics(pred, gt):
    """Paired RGB fidelity plus separated luminance/chroma diagnostics."""
    import cv2
    ps = v15.linear_to_srgb(pred)[0].permute(1,2,0).cpu().numpy().astype(np.float32)
    gs = v15.linear_to_srgb(gt)[0].permute(1,2,0).cpu().numpy().astype(np.float32)
    lab_p = cv2.cvtColor(np.ascontiguousarray(ps), cv2.COLOR_RGB2LAB)
    lab_g = cv2.cvtColor(np.ascontiguousarray(gs), cv2.COLOR_RGB2LAB)
    delta = lab_p-lab_g
    ab = np.sqrt(np.sum(delta[...,1:]**2, axis=2)).mean()
    return {
        'psnr_srgb': float(-10*math.log10(max(np.mean((ps-gs)**2),1e-12))),
        'ssim_srgb': float(ssim_srgb(ps,gs)),
        'lab_ab_error': float(ab),
        'delta_e76': float(np.sqrt(np.sum(delta**2,axis=-1)).mean()),
        'linear_luma_mse': float(((v15.luminance(pred)-v15.luminance(gt))**2).mean()),
        'linear_rgb_L1': float((pred-gt).abs().mean()),
    }


def to_pil_srgb(img_linear):
    p = v15.linear_to_srgb(img_linear)
    a = p[0].permute(1,2,0).cpu().numpy()
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5))


def save_gallery(rows, size, dest):
    if not rows:
        return
    names = [n for n, _ in rows[0][1]]
    title_h, row_gap = 26, 4
    width = size*len(names)
    output = Image.new('RGB', (width, (size+title_h+row_gap)*len(rows)), 'white')
    drawer = ImageDraw.Draw(output)
    for i, (budget, images) in enumerate(rows):
        y = i*(size+title_h+row_gap)
        for j, (name, image) in enumerate(images):
            drawer.text((j*size+2,y+5), f'{name} B={budget:g}', fill=(0,0,0))
            output.paste(image, (j*size,y+title_h))
    output.save(dest)


def eval_main(args):
    if not args.data.is_dir():
        raise FileNotFoundError(f'Dataset directory missing: {args.data}')
    device = v15.make_device(args)
    ds = v15.RGBImages(args.data,args.split,args.size,args.seed,args.val_count)
    args.out.mkdir(parents=True, exist_ok=True)
    requested=set(args.methods)
    required_v15 = requested & LEGACY_METHODS
    if 'ddcolor_full_luma' in requested:
        required_v15.add('full')
    models = {m: load_v15(args,device,m) for m in sorted(required_v15)}
    ddcolor = None
    ck_source = None
    if requested & DDCOLOR_METHODS:
        ddcolor, ck_source = load_ddcolor(args,device)
    print(f'Split={args.split}, n_images={len(ds)}, device={device}; methods={args.methods}',flush=True)
    rows=[]
    gallery = args.out / ('gallery_'+args.split)
    gallery.mkdir(exist_ok=True)
    n=len(ds) if args.eval_images<=0 else min(args.eval_images,len(ds))
    with torch.inference_mode():
        for idx in range(n):
            gt,path=ds[idx]
            gt=gt[None].to(device)
            gt_y=v15.luminance(gt)
            oracle = ddcolorize(ddcolor, gt_y, device) if 'ddcolor_oracle' in requested else None
            gal_rows=[]
            for budget in args.budgets:
                # Exactly V15 evaluation's photon seed and forward model.
                v15.seed_photons(args,idx,budget,salt=300000)
                full,k,est,var,bf=v15.make_features(gt,budget,args,variant=None)
                outputs={}
                for name,model in models.items():
                    outputs[name]=model(make_variant(full,name))
                if 'ddcolor_mle_bf' in requested:
                    outputs['ddcolor_mle_bf']=ddcolorize(ddcolor,bf.clamp(0,1),device)
                if oracle is not None:
                    outputs['ddcolor_oracle']=oracle
                if 'ddcolor_full_luma' in requested:
                    outputs['ddcolor_full_luma']=ddcolorize(
                        ddcolor,v15.luminance(outputs['full']),device)
                for name in args.methods:
                    scores=compute_metrics(outputs[name],gt)
                    rows.append({
                        'scene':str(path), 'budget_total':budget,
                        'cycles':args.cycles,'mode':args.mode,
                        'expected_signal_photons_scene_mean':budget*float(gt_y.mean()),
                        'actual_mean_detections':float(k.mean()),
                        'method':name,'is_oracle':int(name=='ddcolor_oracle'),
                        **scores,
                    })
                if idx<args.gallery_n:
                    cols=[('Counts', to_pil_srgb((k/3.).clamp(0,1))),
                          ('MLE',to_pil_srgb(est.clamp(0,1))),
                          ('MLE+BF',to_pil_srgb(bf.clamp(0,1)))]
                    for name in args.methods:
                        cols.append((name,to_pil_srgb(outputs[name])))
                    cols.append(('RGB GT',to_pil_srgb(gt)))
                    gal_rows.append((budget,cols))
            if idx<args.gallery_n:
                save_gallery(gal_rows,args.size,gallery/f'{idx:04d}.png')
            if (idx+1)%20==0 or idx+1==n:
                print(f'Evaluated {idx+1}/{n}',flush=True)
    out_csv=args.out/f'{args.split}_metrics.csv'
    with out_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)
    out_summary=[]
    for budget in args.budgets:
        for method in args.methods:
            rr=[r for r in rows if r['budget_total']==budget and r['method']==method]
            summ={'budget_total':budget,'method':method,'images':len(rr)}
            for m in ('psnr_srgb','ssim_srgb','lab_ab_error','delta_e76','linear_luma_mse'):
                summ[m]=float(np.mean([r[m] for r in rr]))
            out_summary.append(summ)
            print(f'B={budget:<5g} {method:<20s} '
                  f'PSNR={summ["psnr_srgb"]:6.2f}dB '
                  f'SSIM={summ["ssim_srgb"]:.3f} '
                  f'ab={summ["lab_ab_error"]:.2f}', flush=True)
    summary_csv=args.out/f'{args.split}_summary.csv'
    with summary_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out_summary[0].keys()))
        w.writeheader();w.writerows(out_summary)
    meta={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    meta['ddcolor_checkpoint_source']=ck_source
    (args.out/'config_v16.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'Saved {out_csv} | {summary_csv} | {gallery}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/imagenette2-160'))
    p.add_argument('--v15-out',type=Path,default=Path('runs/v15_64'))
    p.add_argument('--out',type=Path,default=Path('runs/v16_compare'))
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--mode',choices=['bernoulli','poisson'],default='bernoulli')
    p.add_argument('--cycles',type=int,default=64)
    p.add_argument('--budgets',nargs='+',type=float,default=[.5,1,2,4,8])
    p.add_argument('--bg',type=float,default=.01)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--val-count',type=int,default=500)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--bf-radius',type=int,default=3)
    p.add_argument('--bf-sigma',type=float,default=1.5)
    p.add_argument('--bf-knoise',type=float,default=2.0)
    p.add_argument('--bf-prec-cap',type=float,default=4.)
    p.add_argument('--methods',nargs='+',choices=ALL_METHODS,
                   default=list(ALL_METHODS),help='V15 methods plus DDColor. Oracle uses GT luminance.')
    p.add_argument('--ddcolor-repo',type=Path,default=Path('external/DDColor'))
    p.add_argument('--ddcolor-model',type=str,default='piddnad/ddcolor_paper_tiny',
                   help='HF model identifier. Downloaded only if DDColor methods selected.')
    p.add_argument('--ddcolor-weights',type=Path,default=None,
                   help='Local official DDColor weights; bypass HF download')
    p.add_argument('--ddcolor-model-size',choices=['tiny','large'],default='tiny',
                   help='Only for --ddcolor-weights')
    p.add_argument('--ddcolor-size',type=int,default=256,
                   help='DDColor internal colourisation size; 256 (fast) or 512 (higher quality)')
    p.add_argument('--eval-images',type=int,default=100,help='0=all of split')
    p.add_argument('--gallery-n',type=int,default=6)
    args=p.parse_args()
    if args.size<8 or args.size%4!=0: p.error('--size must be divisible by 4, >=8')
    if args.cycles<1 or args.bg<0 or any(b<=0 for b in args.budgets):
        p.error('Require --cycles >=1, --bg>=0 and positive budgets')
    if args.ddcolor_size<32 or args.ddcolor_size%32!=0:
        p.error('--ddcolor-size must be a multiple of 32 and >=32')
    if args.gallery_n<0 or args.eval_images<0: p.error('gallery-n/eval-images must be non-negative')
    if len(set(args.methods))!=len(args.methods): p.error('Methods should not be repeated')
    eval_main(args)

if __name__ == '__main__':
    main()
