#!/usr/bin/env python3
"""V15: static 64x binary SPAD -> one RGB image (single output frame).

Uses v13-inspired Bernoulli likelihood inversion with global-rate shrinkage,
CRLB plug-in variance and precision-weighted bilateral filtering. Three equal-
architecture RGB U-Nets compare raw counts, MLE/BF, and all photon features.
No pretrained color prior, no ground truth at inference.

Command examples: see V15_README.md (Windows PowerShell compatible).
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
VARIANTS = ('raw', 'mle', 'full')


def find_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXT) if folder.is_dir() else []


class RGBImages(Dataset):
    def __init__(self, root, split, size, seed, val_count=500):
        root = Path(root)
        if split == 'train':
            paths = find_images(root / 'train')
        elif split == 'test' and (root / 'test').is_dir():
            paths = find_images(root / 'test')
        else:
            pool = find_images(root / 'val')
            random.Random(seed).shuffle(pool)
            # One deterministic split of original Imagenette val for both runs.
            n_val = min(val_count, max(1, len(pool) // 2))
            paths = pool[:n_val] if split == 'val' else pool[n_val:]
        if not paths:
            raise FileNotFoundError(
                f'No images for {split} under {root.resolve()}. '
                'Expected data/imagenette2-160/train/<class>/*.JPEG and val/<class>/*.JPEG')
        self.paths, self.split, self.size, self.seed = paths, split, size, seed
        self.epoch = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert('RGB')
            w, h = im.size
            r = random.Random(self.seed + 1000003 * self.epoch + i)
            if self.split == 'train':
                side = max(1, int(min(w, h) * r.uniform(0.8, 1.0)))
                left, top = r.randint(0, w - side), r.randint(0, h - side)
            else:
                side = min(w, h)
                left, top = (w - side) // 2, (h - side) // 2
            im = im.crop((left, top, left + side, top + side))
            im = im.resize((self.size, self.size), Image.Resampling.BICUBIC)
            if self.split == 'train' and r.random() < .5:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            a = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1) / 255.
        return srgb_to_linear(a), str(self.paths[i])


def srgb_to_linear(x):
    return torch.where(x <= .04045, x / 12.92, ((x + .055) / 1.055).clamp_min(0) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= .0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - .055)


def luminance(rgb):
    return .2126 * rgb[:, 0:1] + .7152 * rgb[:, 1:2] + .0722 * rgb[:, 2:3]


@torch.no_grad()
def observe(rgb, budget, bg, cycles, mode='bernoulli'):
    """A *single aggregated count image* from T binary exposures of a STATIC scene.

    budget = TOTAL expected signal photons if linear Y=1, across all cycles.
    bg = TOTAL expected background photons across all cycles.
    """
    y = luminance(rgb)
    if mode == 'bernoulli':
        a = budget / cycles
        b = bg / cycles
        p = (-torch.expm1(-(a * y + b))).clamp(1e-10, 1 - 1e-7)
        k = torch.binomial(torch.full_like(p, float(cycles)), p)
        # Claude v13-inspired global-rate empirical-Bayes smoothing.
        # IMPORTANT: unlike fixed Jeffreys +0.5, use scene's observed hit rate.
        m = (k.mean(dim=(-2, -1), keepdim=True) / cycles).clamp(1e-4, 1 - 1e-4)
        smoothing = 1.0
        phat = ((k + smoothing * m) / (cycles + smoothing)).clamp(1e-7, 1 - 1e-7)
        estimate = (-torch.log1p(-phat) - b).clamp_min(0) / max(a, 1e-9)
        # Inverse Fisher information; plug-in finite-count approximation only.
        var = phat / ((cycles * a * a) * (1 - phat)).clamp_min(1e-12)
    elif mode == 'poisson':
        mu = budget * y + bg
        k = torch.poisson(mu)
        estimate = ((k - bg) / max(budget, 1e-9)).clamp_min(0)
        var = (k + .5) / max(budget * budget, 1e-12)
    else:
        raise ValueError(mode)
    return k.float(), estimate.float(), var.float()


@torch.no_grad()
def precision_bilateral(estimate, variance, radius=3, sigma_s=1.5, k_noise=2., prec_cap=4.):
    """Torch/GPU version of Claude v13 precision-weighted bilateral filter.

    Applies to *linear-light* estimate and an approximate variance (no grad).
    """
    if radius == 0:
        return estimate
    B, C, H, W = estimate.shape
    assert C == 1
    v = variance.clamp_min(1e-10)
    k = 2 * radius + 1
    rp = F.pad(estimate, (radius,) * 4, mode='reflect')
    vp = F.pad(v, (radius,) * 4, mode='reflect')
    neighbors = F.unfold(rp, k).view(B, k*k, H, W)
    nv = F.unfold(vp, k).view(B, k*k, H, W).clamp_min(1e-10)
    offsets = torch.arange(-radius, radius+1, device=estimate.device, dtype=estimate.dtype)
    yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
    spatial = torch.exp(-(xx.square() + yy.square()) / (2*sigma_s*sigma_s)).reshape(1, k*k, 1, 1)
    weights = spatial * torch.exp(-((neighbors-estimate)**2) / (2*k_noise*k_noise*(v+nv)))
    weights = weights * (v / nv).clamp(max=prec_cap)
    return (weights * neighbors).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(1e-10)


@torch.no_grad()
def make_features(rgb, budget, args, variant=None):
    k, est, var = observe(rgb, budget, args.bg, args.cycles, args.mode)
    filtered = precision_bilateral(est, var, args.bf_radius, args.bf_sigma,
                                   args.bf_knoise, args.bf_prec_cap)
    # Reflectance in [0,1] is a simulation convention; preserve unclipped
    # estimate for BF and variance, but clip features to calibrated range.
    mle = est.clamp(0, 1)
    bf = filtered.clamp(0, 1)
    # 5 identical input channels for every variant: counts, MLE, BF, variance, exposure.
    raw = (torch.log1p(k) / math.log1p(max(2, args.cycles))).clamp(0, 1)
    uncertainty = (torch.log1p(var) / math.log1p(100.)).clamp(0, 1)
    exposure = torch.full_like(raw, math.log1p(budget) / math.log1p(32.))
    feats = torch.cat([raw, mle, bf, uncertainty, exposure], dim=1)
    if variant == 'raw':
        feats[:, 1:4] = 0
    elif variant == 'mle':
        feats[:, 0:1] = 0
        feats[:, 3:4] = 0
    elif variant == 'full' or variant is None:
        pass
    else:
        raise ValueError(variant)
    return feats, k, est, var, filtered


class ResidualBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1),
            nn.GroupNorm(8, channels_out), nn.SiLU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1),
            nn.GroupNorm(8, channels_out), nn.SiLU())
        self.skip = nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, x):
        return self.body(x) + self.skip(x)


class RGBUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1 = ResidualBlock(5, base)
        self.e2 = ResidualBlock(base, 2*base)
        self.mid = ResidualBlock(2*base, 4*base)
        self.d2 = ResidualBlock(6*base, 2*base)
        self.d1 = ResidualBlock(3*base, base)
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.mid(F.avg_pool2d(b, 2))
        d = self.d2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        e = self.d1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        return torch.sigmoid(self.out(e))   # linear RGB in [0,1]


def compute_loss(pred, gt, k, budget, args):
    l_rgb = F.l1_loss(pred, gt)
    yp, yg = luminance(pred), luminance(gt)
    l_lum = F.l1_loss(yp, yg)
    if args.mode == 'bernoulli':
        p = (-torch.expm1(-(budget / args.cycles * yp + args.bg / args.cycles))).clamp(1e-7, 1 - 1e-7)
        nll = -(k * torch.log(p) + (args.cycles-k) * torch.log1p(-p))
    else:
        mu = budget*yp+args.bg
        nll = mu - k*torch.log(mu.clamp_min(1e-8))
    # One aggregated count image per pixel; rescale for comparable budgets.
    l_ph = nll.mean() / max(budget, 0.1)
    return l_rgb + args.lambda_y*l_lum + args.lambda_ph*l_ph, (l_rgb, l_lum, l_ph)


def seed_photons(args, index, budget, salt=0):
    # Seeds independent of variant, same underlying observations for comparisons.
    torch.manual_seed(args.seed + 1_100_003 + index*7919 + int(round(budget*10000)) + salt)


def psnr_srgb(pred, gt):
    mse = ((linear_to_srgb(pred) - linear_to_srgb(gt)) ** 2).flatten(1).mean(1)
    return -10. * torch.log10(mse.clamp_min(1e-12))


def tensor_image(x):
    return linear_to_srgb(x).expand(-1, 3, -1, -1).clamp(0, 1)


@torch.no_grad()
def monitor(writer, model, dv, variant, args, device, epoch):
    model.eval()
    n = min(args.tb_val_images, len(dv))
    for budget in args.budgets:
        scores = []
        images = []
        for idx in range(n):
            gt, _ = dv[idx]
            gt = gt[None].to(device)
            seed_photons(args, idx, budget, salt=100000)
            feat, k, est, var, den = make_features(gt, budget, args, variant)
            pred = model(feat)
            scores.append(float(psnr_srgb(pred, gt)[0]))
            if len(images) < args.tb_vis_images:
                count_viz = (k / 3).clamp(0, 1).expand(-1, 3, -1, -1)
                images.append(torch.cat([
                    count_viz[0], tensor_image(est.clamp(0,1))[0],
                    tensor_image(den.clamp(0,1))[0],
                    linear_to_srgb(pred)[0], linear_to_srgb(gt)[0]], dim=2).cpu())
        writer.add_scalar(f'PSNR_sRGB/budget_{budget:g}', float(np.mean(scores)), epoch)
        if images:
            writer.add_image(f'RGB/budget_{budget:g}_counts_MLE_BF_prediction_GT',
                             torch.cat(images, dim=1), epoch, dataformats='CHW')
    writer.flush()


def check_data(args):
    print('Dataset:', args.data.resolve())
    for name in ('train','val','test'):
        data = find_images(args.data/name)
        print(f'  {name}: {len(data)} RGB files')
        if data:
            print('  example:', data[0])
    if not find_images(args.data/'train') or not find_images(args.data/'val'):
        raise FileNotFoundError('Imagenette train/val images missing: check --data path')


def make_device(args):
    if args.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Check PyTorch CUDA installation.')
    return torch.device(args.device)


def model_file(args, variant):
    return args.out / (variant + '_best.pt')


def train(args):
    device = make_device(args)
    train_ds = RGBImages(args.data, 'train', args.size, args.seed, args.val_count)
    val_ds = RGBImages(args.data, 'val', args.size, args.seed, args.val_count)
    assert not set(train_ds.paths).intersection(val_ds.paths)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'Device={device} train={len(train_ds)} val={len(val_ds)} mode={args.mode} cycles={args.cycles}', flush=True)
    (args.out/'config.json').write_text(json.dumps({k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, indent=2), encoding='utf-8')
    SummaryWriter = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError('Install tensorboard: python -m pip install tensorboard') from exc
    for variant in args.variants:
        torch.manual_seed(args.seed)
        model = RGBUNet(args.base).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        writer = SummaryWriter(str(args.out/'tb'/variant)) if SummaryWriter else None
        best = float('inf')
        try:
            for epoch in range(args.epochs):
                model.train()
                train_ds.epoch = epoch
                loader_rng = torch.Generator().manual_seed(args.seed + epoch)
                loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                    num_workers=args.workers, generator=loader_rng,
                                    pin_memory=(device.type=='cuda'))
                total, count = 0., 0
                components = np.zeros(3)
                for j, (gt, _) in enumerate(loader):
                    gt = gt.to(device, non_blocking=(device.type=='cuda'))
                    budget = args.budgets[(epoch*1009+j) % len(args.budgets)]
                    seed_photons(args, j+epoch*100000, budget)
                    feat, k, _, _, _ = make_features(gt, budget, args, variant)
                    pred = model(feat)
                    loss, terms = compute_loss(pred, gt, k, budget, args)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    batch_n = gt.size(0)
                    total += float(loss.detach())*batch_n
                    components += np.array([float(t.detach()) for t in terms]) * batch_n
                    count += batch_n
                    if writer and j % args.tb_batch_every == 0:
                        writer.add_scalar('Step/train_total_loss', float(loss.detach()), epoch*len(loader)+j)
                # Deterministic val subset & photon observations for model selection.
                model.eval()
                val_sum, val_n = 0., 0
                with torch.no_grad():
                    for idx in range(min(args.val_images, len(val_ds))):
                        gt, _ = val_ds[idx]
                        gt = gt[None].to(device)
                        budget = args.budgets[idx % len(args.budgets)]
                        seed_photons(args, idx, budget, salt=200000)
                        feat, _, _, _, _ = make_features(gt, budget, args, variant)
                        val_sum += float(F.l1_loss(model(feat), gt))
                        val_n += 1
                val_l1 = val_sum / val_n
                print(f'{variant:>4} epoch {epoch+1:>2}/{args.epochs} '
                      f'train={total/count:.5f} val_RGB_linear_L1={val_l1:.5f}', flush=True)
                if writer:
                    writer.add_scalar('Epoch/train_total_loss', total/count, epoch+1)
                    writer.add_scalar('Epoch/train_RGB_linear_L1', components[0]/count, epoch+1)
                    writer.add_scalar('Epoch/train_luminance_L1', components[1]/count, epoch+1)
                    writer.add_scalar('Epoch/train_photon_NLL_normalised', components[2]/count, epoch+1)
                    writer.add_scalar('Epoch/val_RGB_linear_L1', val_l1, epoch+1)
                    if (epoch+1) % args.tb_every == 0:
                        monitor(writer, model, val_ds, variant, args, device, epoch+1)
                    writer.flush()
                if val_l1 < best:
                    best = val_l1
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch+1,
                                'variant': variant, 'size':args.size, 'base': args.base,
                                'mode':args.mode, 'cycles':args.cycles,
                                'budgets':args.budgets, 'bg':args.bg,
                                'bf_radius':args.bf_radius, 'bf_sigma':args.bf_sigma,
                                'bf_knoise':args.bf_knoise, 'bf_prec_cap':args.bf_prec_cap,
                                'seed':args.seed, 'val_count':args.val_count}, model_file(args, variant))
        finally:
            if writer:
                writer.close()


def save_rgb(x, path):
    x = x.detach().float().cpu().clamp(0,1)
    if x.size(0) == 1:
        x = x.expand(3, -1, -1)
    img = np.uint8(np.clip(x.permute(1,2,0).numpy(), 0, 1)*255+.5)
    Image.fromarray(img).save(path)


@torch.no_grad()
def evaluate(args):
    device = make_device(args)
    ds = RGBImages(args.data, args.split, args.size, args.seed, args.val_count)
    args.out.mkdir(parents=True, exist_ok=True)
    model_list = {}
    for variant in args.variants:
        path = model_file(args, variant)
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint {path}; train first')
        try:
            ck = torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # PyTorch < 2.0 compatibility
            ck = torch.load(path, map_location=device)
        for key in ('mode', 'cycles', 'bg', 'size', 'bf_radius', 'bf_sigma', 'bf_knoise', 'bf_prec_cap','seed','val_count'):
            if ck[key] != getattr(args, key):
                raise ValueError(f'{variant} checkpoint {key}={ck[key]} vs requested {getattr(args,key)}')
        model = RGBUNet(ck['base']).to(device)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        model_list[variant] = model
    rows = []
    gal_dir = args.out / ('gallery_' + args.split)
    gal_dir.mkdir(exist_ok=True)
    n = len(ds) if args.eval_images <= 0 else min(len(ds), args.eval_images)
    for idx in range(n):
        gt, image_path = ds[idx]
        gt = gt[None].to(device)
        scene_gallery = []
        for budget in args.budgets:
            seed_photons(args, idx, budget, salt=300000)
            # One shared observation; features masked per variant.
            full, k, est, var, den = make_features(gt, budget, args)
            outputs = {}
            for variant, net in model_list.items():
                x = full.clone()
                if variant == 'raw':
                    x[:, 1:4] = 0
                elif variant == 'mle':
                    x[:, 0:1] = 0
                    x[:, 3:4] = 0
                output = net(x)
                outputs[variant] = output
                rows.append({'scene':image_path, 'budget_total':budget, 'cycles':args.cycles,
                             'expected_signal_photons_scene_mean':budget*float(luminance(gt).mean()),
                             'actual_mean_detections':float(k.mean()), 'variant':variant,
                             'psnr_srgb':float(psnr_srgb(output, gt)[0]),
                             'linear_luma_mse':float(((luminance(output)-luminance(gt))**2).mean()),
                             'linear_rgb_L1':float(F.l1_loss(output,gt))})
            if idx < args.gallery_n:
                cols = [('Counts', (k/3).clamp(0,1).expand(-1,3,-1,-1)[0]),
                        ('MLE', tensor_image(est.clamp(0,1))[0]),
                        ('MLE+BF', tensor_image(den.clamp(0,1))[0])]
                for variant in args.variants:
                    cols.append((variant, linear_to_srgb(outputs[variant])[0]))
                cols.append(('RGB GT', linear_to_srgb(gt)[0]))
                scene_gallery.append((budget, cols))
        if scene_gallery:
            H = args.size
            ncols = len(scene_gallery[0][1])
            canvas = Image.new('RGB', (ncols*H, len(scene_gallery)*(H+25)), 'white')
            draw = ImageDraw.Draw(canvas)
            for r,(budget,cols) in enumerate(scene_gallery):
                for c,(name,arr) in enumerate(cols):
                    x, y = c*H, r*(H+25)
                    draw.text((x+3,y+4), f'{name} (B={budget:g})', fill='black')
                    im = Image.fromarray(np.uint8(np.clip(arr.float().cpu().permute(1,2,0).numpy(),0,1)*255+.5))
                    canvas.paste(im,(x,y+25))
            canvas.save(gal_dir / f'{idx:04d}.png')
        if (idx+1)%20 == 0 or idx+1 == n:
            print(f'Evaluated {idx+1}/{n}', flush=True)
    output_csv = args.out / f'{args.split}_metrics.csv'
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for budget in args.budgets:
        for variant in args.variants:
            psnr = np.array([r['psnr_srgb'] for r in rows if r['budget_total']==budget and r['variant']==variant])
            print(f'{args.split} total-budget={budget:g} {variant:>4}: PSNR {psnr.mean():.3f} dB, n={len(psnr)}', flush=True)
    print(f'Saved: {output_csv} | galleries: {gal_dir}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train','eval','check-data'])
    parser.add_argument('--data', type=Path, default=Path('data/imagenette2-160'))
    parser.add_argument('--out', type=Path, default=Path('runs/v15_64'))
    parser.add_argument('--mode', choices=['bernoulli','poisson'], default='bernoulli')
    parser.add_argument('--cycles', type=int, default=64)
    parser.add_argument('--budgets', nargs='+', type=float, default=[.5,1,2,4,8])
    parser.add_argument('--bg', type=float, default=.01,
                        help='Total expected background photons per pixel over all exposures')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--base', type=int, default=32)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--workers', type=int, default=0,
                        help='Keep 0 on Windows initially')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda-y', type=float, default=.5)
    parser.add_argument('--lambda-ph', type=float, default=.02)
    parser.add_argument('--bf-radius', type=int, default=3)
    parser.add_argument('--bf-sigma', type=float, default=1.5)
    parser.add_argument('--bf-knoise', type=float, default=2.0)
    parser.add_argument('--bf-prec-cap', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--val-count', type=int, default=500)
    parser.add_argument('--val-images', type=int, default=100,
                        help='Fixed validation subset for checkpoint selection')
    parser.add_argument('--eval-images', type=int, default=0, help='0 = all images')
    parser.add_argument('--split', choices=['val','test'], default='val')
    parser.add_argument('--gallery-n', type=int, default=6)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--tb-val-images', type=int, default=8)
    parser.add_argument('--tb-vis-images', type=int, default=2)
    parser.add_argument('--tb-every', type=int, default=1)
    parser.add_argument('--tb-batch-every', type=int, default=50)
    args = parser.parse_args()
    if args.size % 4 or args.size < 8:
        parser.error('--size must be divisible by 4 and >=8')
    if args.cycles < 1 or any(x <= 0 for x in args.budgets) or args.bg < 0:
        parser.error('Require cycles>=1, budgets>0, bg>=0')
    if args.bf_radius < 0 or args.bf_radius >= args.size:
        parser.error('--bf-radius must be >=0 and smaller than --size')
    if args.val_images < 1 or args.tb_val_images < 1 or args.tb_every < 1:
        parser.error('val-images/tb-val-images/tb-every must be >=1')
    args.data = args.data.expanduser().resolve()
    if args.action == 'check-data':
        check_data(args)
    elif args.action == 'train':
        check_data(args)
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
