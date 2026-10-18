#!/usr/bin/env python3
"""V17: learned photon-to-luminance adapter + frozen DDColor RGB benchmark.

Requires v17_v15_compat.py and v17_v16_compat.py supplied in the V17 package.
Your original train_v15.py and eval_v16.py are NOT imported or overwritten.
- train: supervised luminance adapter (training RGB GT used only to form Y target).
- eval: same V15/V16 photon draws and all six legacy methods, plus new adapter
  grayscale diagnostic and adapter -> pretrained DDColor RGB.

This is NOT a latent bridge. No test/val GT is supplied to non-oracle inference.
See V17_README.md for commands and research caveats.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

import v17_v15_compat as v15
import v17_v16_compat as v16

LEGACY = v16.ALL_METHODS
ALL_METHODS = LEGACY + ('ddcolor_adapter',)


def make_device(args):
    """V17-owned device resolver; avoid requiring v15.make_device()."""
    wanted = args.device
    if wanted == 'auto':
        wanted = 'cuda' if torch.cuda.is_available() else 'cpu'
    if wanted == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA requested but unavailable in this Python environment. '
            'Check python -c "import torch; print(torch.cuda.is_available())"'
        )
    return torch.device(wanted)


def check_v15_api():
    """Report all missing V15 symbols at once rather than failing mid-training."""
    required = ('RGBImages', 'ResidualBlock', 'RGBUNet', 'luminance',
                'linear_to_srgb', 'srgb_to_linear', 'seed_photons', 'make_features')
    missing = [name for name in required if not hasattr(v15, name)]
    if missing:
        raise RuntimeError(
            f'Incompatible bundled v17_v15_compat.py at {getattr(v15, "__file__", "unknown")}: '
            f'missing {", ".join(missing)}. This V17 implementation expects '
            'the V15 model/features version used by your V16 evaluation. '
            'Reinstall the complete V17 bundle with the two compatibility files.'
        )


class LumaAdapter(nn.Module):
    """5-channel photon features -> one linear-light luminance channel in [0,1]."""
    def __init__(self, base=32):
        super().__init__()
        self.e1 = v15.ResidualBlock(5, base)
        self.e2 = v15.ResidualBlock(base, base*2)
        self.mid = v15.ResidualBlock(base*2, base*4)
        self.d2 = v15.ResidualBlock(base*6, base*2)
        self.d1 = v15.ResidualBlock(base*3, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.mid(F.avg_pool2d(b, 2))
        d = self.d2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], dim=1))
        e = self.d1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], dim=1))
        return torch.sigmoid(self.out(e))


def poisson_or_binomial_nll(y_hat, k, budget, args):
    """NLL of the *actual counts* under the V15 forward model; differentiable."""
    if args.mode == 'bernoulli':
        q = -torch.expm1(-(budget/args.cycles*y_hat + args.bg/args.cycles))
        q = q.clamp(1e-7, 1-1e-7)
        loss = -(k*torch.log(q) + (args.cycles-k)*torch.log1p(-q))
    else:
        mu = (budget*y_hat + args.bg).clamp_min(1e-8)
        loss = mu - k*torch.log(mu)  # omit k! independent of model
    return loss.mean() / max(budget, 0.1)


def edge_l1(pred, target):
    return (pred[...,1:,:]-pred[...,:-1,:]).sub(target[...,1:,:]-target[...,:-1,:]).abs().mean() + \
           (pred[...,:,1:]-pred[...,:,:-1]).sub(target[...,:,1:]-target[...,:,:-1]).abs().mean()


def luma_loss(yhat, ygt, k, budget, args):
    a = F.l1_loss(yhat, ygt)
    e = edge_l1(yhat, ygt)
    ph = poisson_or_binomial_nll(yhat, k, budget, args)
    return a + args.lambda_edge*e + args.lambda_ph*ph, (a, e, ph)


def adapter_path(args):
    return args.out / 'luma_adapter_best.pt'


def configs_for_checkpoint(args):
    keys = ('mode','cycles','bg','size','bf_radius','bf_sigma', 'bf_knoise',
            'bf_prec_cap','seed','val_count')
    return {key:getattr(args,key) for key in keys}


def check_checkpoint(ck, args):
    for key, value in configs_for_checkpoint(args).items():
        if ck.get(key) != value:
            raise ValueError(f'Adapter mismatch: {key} checkpoint={ck.get(key)} vs CLI={value}. '
                             f'Use original acquisition settings.')


def load_adapter(args, device):
    p = adapter_path(args)
    if not p.is_file():
        raise FileNotFoundError(f'Missing {p}; first run the train action.')
    try:
        ck = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(p, map_location=device)
    check_checkpoint(ck, args)
    net = LumaAdapter(ck['base']).to(device)
    net.load_state_dict(ck['state_dict'], strict=True)
    net.eval()
    print(f'Loaded adapter checkpoint: {p} (best epoch={ck["epoch"]})', flush=True)
    return net


def choose_training_budget(args, epoch, step):
    # All exposure conditions repeated in stable order across epochs.
    return args.budgets[(epoch * 101 + step) % len(args.budgets)]


@torch.no_grad()
def validate_adapter(net, ds, args, device, epoch):
    net.eval()
    sums = {float(b): [0.,0.,0] for b in args.budgets}  # mae, mse, count
    samples = []
    for idx in range(min(args.val_images,len(ds))):
        gt,_ = ds[idx]
        gt = gt[None].to(device)
        y_gt = v15.luminance(gt)
        for b in args.budgets:
            # Validation seeds held constant across training epochs (separate from eval).
            v15.seed_photons(args, idx, b, salt=200000)
            feats, k, est, var, bf = v15.make_features(gt,b,args)
            y_hat = net(feats)
            sums[float(b)][0] += float(F.l1_loss(y_hat,y_gt))
            sums[float(b)][1] += float(F.mse_loss(y_hat,y_gt))
            sums[float(b)][2] += 1
            if idx == 0 and len(samples)<2 and b in (args.budgets[0],args.budgets[-1]):
                samples.append((b,k,est,bf,y_hat,y_gt))
    stats={b:{'mae':s[0]/max(1,s[2]),'mse':s[1]/max(1,s[2])} for b,s in sums.items()}
    val_mae = float(np.mean([s['mae'] for s in stats.values()]))
    return val_mae,stats,samples


def check_installation(args):
    """Fail fast on old V15 checkpoint/model compatibility before long GPU training."""
    check_v15_api()
    print('V17 bundled compatibility modules:', flush=True)
    print('  v15:', Path(v15.__file__).resolve(), flush=True)
    print('  v16:', Path(v16.__file__).resolve(), flush=True)
    for split in ('train', 'val'):
        ds = v15.RGBImages(args.data, split, args.size, args.seed, args.val_count)
        print(f'  {split}: {len(ds)} images', flush=True)
    # Load using CPU to avoid wasting CUDA VRAM during the preflight check.
    for variant in ('raw', 'mle', 'full'):
        v16.load_v15(args, torch.device('cpu'), variant)
    print('PASS: dataset, all three V15 checkpoints and bundled APIs are compatible.', flush=True)


def train_main(args):
    check_installation(args)
    device=make_device(args)
    if args.val_images<=0:
        raise ValueError('--val-images should be positive during training')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    ds=v15.RGBImages(args.data,'train',args.size,args.seed,args.val_count)
    dv=v15.RGBImages(args.data,'val',args.size,args.seed,args.val_count)
    if set(map(str,ds.paths)).intersection(map(str,dv.paths)):
        raise RuntimeError('Training/validation leakage detected')
    args.out.mkdir(parents=True,exist_ok=True)
    device_string=str(device)
    print(f'Train adapter: device={device_string} train={len(ds)} val={len(dv)} '
          f'cycles={args.cycles} budgets={args.budgets}; V15 models are NOT retrained.', flush=True)
    model=LumaAdapter(args.base).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    writer=None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise RuntimeError('Install TensorBoard: python -m pip install tensorboard') from e
        writer=SummaryWriter(str(args.out/'tb'/'luma_adapter'))
    best=float('inf')
    train_records=[]
    try:
        for epoch in range(args.epochs):
            model.train()
            ds.epoch=epoch
            dl=DataLoader(ds,batch_size=args.batch,shuffle=True,num_workers=args.workers,
                          pin_memory=device.type=='cuda',
                          generator=torch.Generator().manual_seed(args.seed+epoch))
            total=0.;totals=np.zeros(3);n=0
            for step,(gt,_) in enumerate(dl):
                gt=gt.to(device,non_blocking=device.type=='cuda')
                budget=choose_training_budget(args,epoch,step)
                # Exposure reproducible independent of DataLoader global RNG and model updates.
                v15.seed_photons(args,step+epoch*100000,budget)
                feats,k,_,_,_=v15.make_features(gt,budget,args)
                y_gt=v15.luminance(gt)
                y_hat=model(feats)
                loss,terms=luma_loss(y_hat,y_gt,k,budget,args)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip>0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
                opt.step()
                ni=len(gt)
                n+=ni
                total+=float(loss.detach())*ni
                totals+=np.array([float(t.detach()) for t in terms])*ni
                if writer and step%args.tb_batch_every==0:
                    writer.add_scalar('Step/train_total',float(loss.detach()),epoch*len(dl)+step)
            score,stats,samples=validate_adapter(model,dv,args,device,epoch)
            print(f'Adapter epoch {epoch+1}/{args.epochs}: train={total/n:.5f} '
                  f'val_linear_luma_L1={score:.5f} '
                  + ' '.join(f'B{b:g}:{stats[float(b)]["mae"]:.4f}' for b in args.budgets),flush=True)
            rec={'epoch':epoch+1,'train_total':total/n,'train_luma_L1':totals[0]/n,
                 'train_edge_L1':totals[1]/n,'train_photon_NLL':totals[2]/n,
                 'val_mean_luma_L1':score}
            for b in args.budgets:
                rec[f'val_luma_L1_B{b:g}']=stats[float(b)]['mae']
                rec[f'val_luma_MSE_B{b:g}']=stats[float(b)]['mse']
            train_records.append(rec)
            if writer:
                for name,val in rec.items():
                    if name!='epoch':writer.add_scalar(name,float(val),epoch+1)
                if (epoch+1)%args.tb_every==0:
                    for b,k,est,bf,y_hat,y_gt in samples:
                        # Gray gallery: counts | MLE | BF | adapter | GT, in sRGB display
                        count_view=(k / 3.).clamp(0,1)[0].expand(3,-1,-1)
                        gray=[v15.linear_to_srgb(a.clamp(0,1))[0].expand(3,-1,-1)
                              for a in [est,bf,y_hat,y_gt]]
                        grid=torch.cat([count_view]+gray,dim=2)
                        writer.add_image(f'Gallery/B{b:g}_counts_MLE_BF_adapter_GT',grid.cpu(),epoch+1)
                writer.flush()
            if score<best:
                best=score
                ck={'state_dict':model.state_dict(), 'base':args.base,'epoch':epoch+1,
                    'budgets':list(args.budgets), 'lambda_edge':args.lambda_edge,
                    'lambda_ph':args.lambda_ph, **configs_for_checkpoint(args)}
                torch.save(ck,adapter_path(args))
    finally:
        if writer:writer.close()
    with (args.out/'adapter_train_history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(train_records[0].keys()))
        w.writeheader();w.writerows(train_records)
    print(f'Saved adapter: {adapter_path(args)}',flush=True)


def safe_scalar(t):
    return float(t.detach().cpu().item())


def adapter_luma_metrics(y_hat,ygt):
    return {
        'linear_luma_mse':safe_scalar(F.mse_loss(y_hat,ygt)),
        'linear_luma_mae':safe_scalar(F.l1_loss(y_hat,ygt)),
        'gray_psnr_srgb':safe_scalar(-10*torch.log10(
            (v15.linear_to_srgb(y_hat)-v15.linear_to_srgb(ygt)).square().mean().clamp_min(1e-12)))
    }


def write_csv(path,rows):
    if not rows:raise ValueError('No results to write')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)


@torch.no_grad()
def eval_main(args):
    check_v15_api()
    device=make_device(args)
    ds=v15.RGBImages(args.data,args.split,args.size,args.seed,args.val_count)
    args.out.mkdir(parents=True,exist_ok=True)
    req=set(args.methods)
    load_legacy=(req & v16.LEGACY_METHODS)
    if 'ddcolor_full_luma' in req:load_legacy.add('full')
    legacy_models={m:v16.load_v15(args,device,m) for m in sorted(load_legacy)}
    adapter=load_adapter(args,device)
    colorizer=None
    source=None
    if req & v16.DDCOLOR_METHODS or 'ddcolor_adapter' in req:
        colorizer,source=v16.load_ddcolor(args,device)
    print(f'V17 eval: split={args.split} total={len(ds)} mode={args.mode} '
          f'budgets={args.budgets} methods={args.methods}',flush=True)
    output=[]
    luma_rows=[]
    folder=args.out/('gallery_'+args.split)
    folder.mkdir(exist_ok=True)
    n=len(ds) if args.eval_images==0 else min(args.eval_images,len(ds))
    if n<1:raise ValueError('No evaluation images')
    for idx in range(n):
        gt, path=ds[idx]
        gt=gt[None].to(device)
        y_gt=v15.luminance(gt)
        oracle=v16.ddcolorize(colorizer,y_gt,device) if 'ddcolor_oracle' in req else None
        gallery=[]
        for budget in args.budgets:
            # Same photon measurement seed as V15 and V16 evaluation.
            v15.seed_photons(args,idx,budget,salt=300000)
            feats,k,est,var,bf=v15.make_features(gt,budget,args)
            y_hat=adapter(feats)
            # Diagnostic all methods see same photon observation; GT only used for score/oracle.
            y_scores={
                'adapter':adapter_luma_metrics(y_hat,y_gt),
                'mle':adapter_luma_metrics(est.clamp(0,1),y_gt),
                'mle_bf':adapter_luma_metrics(bf.clamp(0,1),y_gt)
            }
            outputs={}
            for name,net in legacy_models.items():
                outputs[name]=net(v16.make_variant(feats,name))
            if 'ddcolor_mle_bf' in req:
                outputs['ddcolor_mle_bf']=v16.ddcolorize(colorizer,bf.clamp(0,1),device)
            if oracle is not None:
                outputs['ddcolor_oracle']=oracle
            if 'ddcolor_full_luma' in req:
                outputs['ddcolor_full_luma']=v16.ddcolorize(colorizer,v15.luminance(outputs['full']),device)
            if 'ddcolor_adapter' in req:
                outputs['ddcolor_adapter']=v16.ddcolorize(colorizer,y_hat,device)
            photon_info={
                'scene':str(path),'budget_total':float(budget),'cycles':args.cycles,'mode':args.mode,
                'expected_signal_photons_scene_mean':float(budget*y_gt.mean()),
                'actual_mean_detections':float(k.mean()),
            }
            for method,scores in y_scores.items():
                luma_rows.append({**photon_info,'method':method,**scores})
            if 'full' in outputs:
                luma_rows.append({**photon_info,'method':'v15_full_luma',
                                  **adapter_luma_metrics(v15.luminance(outputs['full']),y_gt)})
            for name in args.methods:
                output.append({**photon_info,'method':name,'is_oracle':int(name=='ddcolor_oracle'),
                               **v16.compute_metrics(outputs[name],gt)})
            if idx<args.gallery_n:
                cols=[('Counts',v16.to_pil_srgb((k/3.).clamp(0,1))),
                      ('MLE',v16.to_pil_srgb(est.clamp(0,1))),
                      ('MLE+BF',v16.to_pil_srgb(bf.clamp(0,1))),
                      ('Adapter Y',v16.to_pil_srgb(y_hat))]
                for name in args.methods:cols.append((name,v16.to_pil_srgb(outputs[name])))
                cols.append(('RGB GT',v16.to_pil_srgb(gt)))
                gallery.append((budget,cols))
        if idx<args.gallery_n:
            v16.save_gallery(gallery,args.size,folder/f'{idx:04d}.png')
        if (idx+1)%20==0 or idx+1==n:
            print(f'Evaluated {idx+1}/{n}',flush=True)
    write_csv(args.out/f'{args.split}_metrics.csv',output)
    write_csv(args.out/f'{args.split}_luma_metrics.csv',luma_rows)
    summary=[]
    for budget in args.budgets:
        for name in args.methods:
            rows=[r for r in output if r['budget_total']==budget and r['method']==name]
            ss={'budget_total':float(budget),'method':name,'images':len(rows)}
            for key in ('psnr_srgb','ssim_srgb','lab_ab_error','delta_e76','linear_luma_mse'):
                ss[key]=float(np.mean([r[key] for r in rows]))
            summary.append(ss)
            print(f'B={budget:g} {name:<22s} RGB PSNR={ss["psnr_srgb"]:.3f} '
                  f'SSIM={ss["ssim_srgb"]:.3f} ab_error={ss["lab_ab_error"]:.2f}',flush=True)
    write_csv(args.out/f'{args.split}_summary.csv',summary)
    luma_summary=[]
    for budget in args.budgets:
        for name in sorted(set(r['method'] for r in luma_rows)):
            rows=[r for r in luma_rows if r['budget_total']==budget and r['method']==name]
            if rows:
                luma_summary.append({'budget_total':float(budget),'method':name,'images':len(rows),
                                     **{k:float(np.mean([r[k] for r in rows]))
                                        for k in ('linear_luma_mse','linear_luma_mae','gray_psnr_srgb')}})
    write_csv(args.out/f'{args.split}_luma_summary.csv',luma_summary)
    print('Luminance diagnostics (linear-Y MSE; lower is better):', flush=True)
    for row in luma_summary:
        print(f'  B={row["budget_total"]:g} {row["method"]:<16s} MSE={row["linear_luma_mse"]:.5f} MAE={row["linear_luma_mae"]:.5f}', flush=True)
    meta={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    meta['ddcolor_checkpoint_source']=source
    (args.out/'config_v17.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'Saved metrics, luma diagnostics, summaries and gallery to {args.out}',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','train','eval'])
    p.add_argument('--data',type=Path,default=Path('data/imagenette2-160'))
    p.add_argument('--v15-out',type=Path,default=Path('runs/v15_64'))
    p.add_argument('--out',type=Path,default=Path('runs/v17_adapter'))
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--mode',choices=['bernoulli','poisson'],default='bernoulli')
    p.add_argument('--cycles',type=int,default=64)
    p.add_argument('--budgets',nargs='+',type=float,default=[.5,1.,2.,4.,8.])
    p.add_argument('--bg',type=float,default=.01)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--val-count',type=int,default=500)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--bf-radius',type=int,default=3)
    p.add_argument('--bf-sigma',type=float,default=1.5)
    p.add_argument('--bf-knoise',type=float,default=2.)
    p.add_argument('--bf-prec-cap',type=float,default=4.)
    p.add_argument('--base',type=int,default=32)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--workers',type=int,default=0,help='For Windows start with 0')
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--lambda-edge',type=float,default=.1)
    p.add_argument('--lambda-ph',type=float,default=.02)
    p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--val-images',type=int,default=50)
    p.add_argument('--eval-images',type=int,default=20,help='0=entire split')
    p.add_argument('--gallery-n',type=int,default=6)
    p.add_argument('--methods',nargs='+',choices=ALL_METHODS,default=list(ALL_METHODS))
    p.add_argument('--tensorboard',action='store_true')
    p.add_argument('--tb-every',type=int,default=1)
    p.add_argument('--tb-batch-every',type=int,default=50)
    p.add_argument('--ddcolor-repo',type=Path,default=Path('external/DDColor'))
    p.add_argument('--ddcolor-model',type=str,default='piddnad/ddcolor_paper_tiny')
    p.add_argument('--ddcolor-weights',type=Path,default=None)
    p.add_argument('--ddcolor-model-size',choices=['tiny','large'],default='tiny')
    p.add_argument('--ddcolor-size',type=int,default=256)
    return p


def main():
    p=parser();args=p.parse_args()
    if args.size<8 or args.size%4:p.error('size must be >=8 and divisible by 4')
    if args.cycles<1 or args.bg<0 or any(x<=0 for x in args.budgets):p.error('cycles>=1 bg>=0 budgets>0')
    if args.bf_radius<0 or args.bf_radius>=args.size:p.error('bf-radius must be >=0 and < size')
    if args.val_images<1 or args.eval_images<0 or args.gallery_n<0:p.error('val-images positive, eval-images/gallery-n >=0')
    if args.ddcolor_size%32 or args.ddcolor_size<32:p.error('ddcolor-size must be multiple of 32')
    if len(set(args.methods))!=len(args.methods):p.error('--methods cannot repeat methods')
    if args.action=='check':check_installation(args)
    elif args.action=='train':train_main(args)
    else:eval_main(args)


if __name__=='__main__':main()
