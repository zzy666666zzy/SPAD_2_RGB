#!/usr/bin/env python3
"""Two-track SPAD RGB benchmark, based on the original quickbench SIC code.

Research reimplementation inspired by Purohit et al., CVPR 2024, NOT official
reproduction / author checkpoint. Active sensor RGB-linear-luminance PROXY,
not physically calibrated NIR reflectance. Requires no new conda environment.

This file REUSES the quickbench Imagenette loader, split logic, dataset class,
linear sRGB functions, and checkpoint-compatible Direct/Residual UNet.
Removes tiny DDPM and the external DDColor subprocess. Each track now trains
its own matched U-Nets, colorizer, AtomODE and PhotonODE. Active NIR input
requires paired NIR/RGB data; otherwise --track active is a visible-luma proxy.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
try:
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.color import rgb2lab
except ImportError:
    sk_ssim = None
    rgb2lab = None

URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
SHA256 = '64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5'

def get_data(root, allow_download=True):
    d = root / 'imagenette2-160'
    if (d / 'train').is_dir() and (d / 'val').is_dir():
        return d
    if not allow_download:
        raise RuntimeError(f'Missing {d}. Remove --no-download or provide --data-root.')
    root.mkdir(parents=True, exist_ok=True)
    arc = root / 'imagenette2-160.tgz'
    if not arc.exists() or arc.stat().st_size < 90_000_000:
        print(f'Downloading ~99MB: {URL}', flush=True)
        urllib.request.urlretrieve(URL, arc)
    h = hashlib.sha256()
    with arc.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    if h.hexdigest() != SHA256:
        raise RuntimeError('Dataset SHA-256 mismatch; delete corrupted archive and retry')
    print('Extracting Imagenette...', flush=True)
    with tarfile.open(arc, 'r:gz') as tf:
        for m in tf:
            p = Path(m.name)
            if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'imagenette2-160':
                raise RuntimeError(f'Unsafe archive member: {m.name}')
            if not (m.isdir() or m.isfile()):
                raise RuntimeError(f'Unsupported archive entry: {m.name}')
            # Python 3.9/3.10 does not accept extract(filter=...). Paths/types
            # were checked above; never follow links or special archive members.
            if hasattr(tarfile, 'data_filter'):
                tf.extract(m, root, filter='data')
            else:
                tf.extract(m, root)
    return d


def images_from(folder):
    return sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.jpeg', '.jpg', '.png', '.webp'))


def split_data(data, tr, va, te, seed):
    train = images_from(data / 'train')
    held = images_from(data / 'val')
    r = random.Random(seed)
    r.shuffle(train)
    r.shuffle(held)
    if len(train) < tr or len(held) < va + te:
        raise ValueError(f'Insufficient images: train={len(train)} heldout={len(held)}')
    return {'train': train[:tr], 'val': held[:va], 'test': held[va:va + te]}


class RGBDataset(torch.utils.data.Dataset):
    def __init__(self, files, size, augment=False, nir_root=None, rgb_root=None):
        self.files, self.size, self.augment = files, size, augment
        self.nir_root, self.rgb_root = nir_root, rgb_root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as f:
            rgb = ImageOps.fit(f.convert('RGB'), (self.size, self.size), method=Image.Resampling.BICUBIC)
        nir = None
        if self.nir_root is not None:
            relative = self.files[i].relative_to(self.rgb_root)
            nir_path = self.nir_root / relative
            if not nir_path.is_file():
                raise FileNotFoundError('Missing paired NIR image: %s' % nir_path)
            with Image.open(nir_path) as f:
                if len(f.getbands()) != 1:
                    raise ValueError('Paired NIR must be single-channel: %s' % nir_path)
                raw_nir = np.asarray(f)
                # Fixed-range conversion, never per-image min-max scaling.
                if raw_nir.dtype == np.uint8:
                    factor = 255.0
                elif raw_nir.dtype == np.uint16:
                    factor = 65535.0
                elif np.issubdtype(raw_nir.dtype, np.floating):
                    factor = 1.0  # Float NIR image must already be in [0,1].
                else:
                    raise ValueError('Unsupported NIR format (%s) for %s' % (raw_nir.dtype, nir_path))
                nir_float = np.clip(raw_nir.astype(np.float32) / factor, 0, 1)
                nir = ImageOps.fit(Image.fromarray(nir_float, mode='F'),
                                   (self.size, self.size), method=Image.Resampling.BICUBIC)
        if self.augment and random.random() < 0.5:
            rgb = ImageOps.mirror(rgb)
            if nir is not None: nir = ImageOps.mirror(nir)
        arr = np.asarray(rgb, dtype=np.float32).copy().transpose(2, 0, 1) / 255.0
        rgb_tensor = torch.from_numpy(arr)
        if nir is None: return rgb_tensor
        # NIR pixel values must be calibrated to normalized linear reflectance.
        nir_array = np.clip(np.asarray(nir, dtype=np.float32), 0, 1).copy()[None]
        return rgb_tensor, torch.from_numpy(nir_array)


def srgb_to_linear(v):
    return torch.where(v <= .04045, v / 12.92, ((v + .055) / 1.055).pow(2.4))


def linear_to_srgb(v):
    v = v.clamp(0, 1)
    return torch.where(v <= .0031308, 12.92 * v, 1.055 * v.pow(1 / 2.4) - .055)


class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(min(8, b), b), nn.SiLU(),
            nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(min(8, b), b), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class UNetRGB(nn.Module):
    def __init__(self, ch=24, residual=False):
        super().__init__()
        self.residual = residual
        self.enc1 = DoubleConv(3, ch)
        self.enc2 = DoubleConv(ch, ch * 2)
        self.enc3 = DoubleConv(ch * 2, ch * 4)
        self.middle = DoubleConv(ch * 4, ch * 4)
        self.dec2 = DoubleConv(ch * 4 + ch * 2, ch * 2)
        self.dec1 = DoubleConv(ch * 2 + ch, ch)
        self.head = nn.Conv2d(ch, 3, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))
        z = self.middle(c)
        z = self.dec2(torch.cat([F.interpolate(z, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        z = self.dec1(torch.cat([F.interpolate(z, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        delta = self.head(z)
        if self.residual:
            return (x[:, 1:2].repeat(1, 3, 1, 1) + .5 * torch.tanh(delta)).clamp(0, 1)
        return torch.sigmoid(delta)



def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_auto(requested):
    if requested != 'auto':
        if requested == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; use --device cpu')
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def split_files(args):
    manifest = args.out / 'manifest.json'
    if args.mode == 'eval':
        if not manifest.exists():
            raise FileNotFoundError(f'{manifest} missing: first run --mode train')
        fs = {k: [Path(p) for p in v] for k, v in json.loads(manifest.read_text()).items()}
        absent = [p for group in fs.values() for p in group if not p.exists()]
        if absent:
            raise FileNotFoundError(f'Manifest image missing: {absent[0]}')
        return fs
    root = get_data(args.data_root, allow_download=not args.no_download)
    fs = split_data(root, args.train_count, args.val_count, args.test_count, args.seed)
    as_json = {k: [str(p) for p in v] for k, v in fs.items()}
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous != as_json:
            raise ValueError('Output directory uses a DIFFERENT data split; use a fresh --out or restore original counts/seed')
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(as_json, indent=2))
    return fs

def reflectance_proxy(rgb):
    y = srgb_to_linear(rgb)
    return .2126*y[:, 0:1] + .7152*y[:, 1:2] + .0722*y[:, 2:3]


def binary_from_y(y, alpha, background=0., gen=None):
    # Bernoulli photon sensor: Pr(Y=1)=1-exp[-(alpha*rho + background)]
    p = -torch.expm1(-(y*float(alpha) + float(background)))
    # sample on CPU for reproducibility and macOS MPS compatibility
    if gen is None:
        z = torch.rand(p.shape, device=p.device)
    else:
        z = torch.rand(p.shape, generator=gen, device=p.device)
    return (z < p).float()


def poisson_from_y(y, alpha, background=0., gen=None):
    """Active laser photon-count proxy, assuming ideal Poisson detections.

    Real pulsed active SPAD may require Binomial/TCSPC, pileup/dead-time etc.
    """
    rate = (y * float(alpha) + float(background)).clamp_min(0.)
    return torch.poisson(rate, generator=gen)


def observation(y, alpha, background=0., gen=None, track='binary'):
    if track == 'binary':
        return binary_from_y(y, alpha, background, gen)
    if track == 'active':
        return poisson_from_y(y, alpha, background, gen)
    raise ValueError('Unknown track: ' + str(track))


def target_expected(y, alpha_target, background=0., track='binary'):
    if track == 'binary':
        return -torch.expm1(-(y*float(alpha_target) + float(background)))
    # The active colorizer receives calibrated normalized signal intensity.
    return y.clamp(0., 1.)


def observation_display(obs, alpha, background, track):
    if track == 'binary': return obs
    # Poisson count image is converted to normalized reflectance proxy for display.
    return ((obs - background)/max(alpha, 1e-6)).clamp(0., 1.)


def colorizer_input(obs, alpha, background, track):
    return observation_display(obs, alpha, background, track)


def matched_unet_condition(obs, alpha, background, track):
    """Same measurement for both networks; conditioned on exposure level.

    3-channel U-Net expects raw, smoothed, exposure plane. Both tracks feed
    through this identical interface, but each U-Net is trained separately.
    """
    raw = observation_display(obs, alpha, background, track)
    smoothed = F.avg_pool2d(raw, 5, 1, 2).clamp(0, 1)
    if track == 'active':
        # The original U-Net conditioning uses gamma-encoded intensity.
        raw = linear_to_srgb(raw)
        smoothed = linear_to_srgb(smoothed)
    strength = torch.full_like(raw, math.log1p(alpha) / math.log1p(40.))
    return torch.cat([raw, smoothed, strength], 1)


def batch_rgb_y(batch, device, track):
    if isinstance(batch, (tuple, list)):
        rgb = batch[0].to(device)
        if track != 'active': raise ValueError('NIR pairs only supported for --track active')
        y = batch[1].to(device)
    else:
        rgb = batch.to(device)
        y = reflectance_proxy(rgb)
    return rgb, y


def features(obs, alpha, alpha_target, variant, background, track='binary'):
    # Both tracks estimate source reflectance and its approximate uncertainty.
    if track == 'binary':
        local = F.avg_pool2d(obs, 5, stride=1, padding=2).clamp(.0001, .999)
        rate = (-torch.log1p(-local) - float(background)).clamp_min(0.)
        rho = (rate / max(float(alpha), 1e-6)).clamp(0., 1.)
        stderr = torch.sqrt((local * (1.-local)).clamp_min(.0001) / 25.)
        sigma_rho = stderr / (max(float(alpha), .05) * (1.-local).clamp_min(.08))
        view = obs
        local_view = local
    else:
        # Local Poisson MLE: E[count] = alpha*rho + background.
        local = F.avg_pool2d(obs, 5, stride=1, padding=2).clamp_min(0.)
        rho = ((local - float(background)) / max(float(alpha), 1e-6)).clamp(0., 1.)
        sigma_rho = torch.sqrt(local.clamp_min(1e-4) / 25.) / max(float(alpha), .05)
        view = observation_display(obs, alpha, background, track)
        local_view = rho
    analytic = target_expected(rho, alpha_target, background, track)
    confidence = torch.exp(-5.0 * sigma_rho).clamp(0., 1.)
    alpha_log = math.log(max(float(alpha), 1e-4)/float(alpha_target))
    alpha_plane = torch.full_like(view, alpha_log/3.)
    if variant == 'photon':
        inp = torch.cat([view, local_view, confidence, alpha_plane], dim=1)
    else:
        inp = torch.cat([view, torch.zeros_like(view),
                         torch.zeros_like(view), alpha_plane], dim=1)
    return inp, analytic, confidence


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(min(4, cout), cout), nn.SiLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(min(4, cout), cout), nn.SiLU())
    def forward(self, x):
        return self.net(x)


class Pix2PixColorizer(nn.Module):
    """Compact U-Net Pix2Pix generator (1-channel fixed exposure -> RGB)."""
    def __init__(self, ch=16):
        super().__init__()
        self.c1 = ConvBlock(1, ch)
        self.c2 = ConvBlock(ch, 2*ch)
        self.c3 = ConvBlock(2*ch, 4*ch)
        self.mid = ConvBlock(4*ch, 4*ch)
        self.up2 = ConvBlock(6*ch, 2*ch)
        self.up1 = ConvBlock(3*ch, ch)
        self.out = nn.Conv2d(ch, 3, 1)
    def forward(self, inp):
        a = self.c1(inp)
        b = self.c2(F.avg_pool2d(a, 2))
        c = self.c3(F.avg_pool2d(b, 2))
        z = self.mid(c)
        z = self.up2(torch.cat([F.interpolate(z, size=b.shape[-2:], mode='bilinear',align_corners=False), b], 1))
        z = self.up1(torch.cat([F.interpolate(z, size=a.shape[-2:], mode='bilinear',align_corners=False), a], 1))
        return torch.sigmoid(self.out(z))


class PatchDiscriminator(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4,ch,4,2,1),nn.LeakyReLU(.2),
                                 nn.Conv2d(ch,2*ch,4,2,1),nn.LeakyReLU(.2),
                                 nn.Conv2d(2*ch,4*ch,4,2,1),nn.LeakyReLU(.2),
                                 nn.Conv2d(4*ch,1,3,1,1))
    def forward(self, gray, rgb):
        return self.net(torch.cat([gray,rgb],1))


class AtomDynamics(nn.Module):
    """Six linear layers define dΛ/dt; Λ contains 2 x m x 3x3 atoms."""
    def __init__(self, n_atoms=6, hidden=64):
        super().__init__()
        self.dimension = 2*n_atoms*9
        layers=[]
        n=self.dimension+1
        for _ in range(5):
            layers += [nn.Linear(n, hidden), nn.SiLU()]
            n=hidden
        layers += [nn.Linear(n, self.dimension)]
        self.net=nn.Sequential(*layers)
    def forward(self, atoms, t):
        tchannel = t.reshape(1).to(atoms).expand(1)
        return self.net(torch.cat([atoms.reshape(-1),tchannel])).reshape_as(atoms)


class AtomODE(nn.Module):
    """Exposure-adaptive filter atom integration, a lightweight paper-inspired implementation.
    Euler/RK4 integration builds 3x3 filter atoms as a continuous function of
    log(source alpha / target alpha). Coefficients map atoms to convolution kernels.
    Not an exact copy of the published AtomODE-Pix2Pix implementation.
    """
    def __init__(self, ch=12, atoms=6, steps=2, variant='atomode'):
        super().__init__()
        self.ch, self.n_atoms, self.steps, self.variant = ch, atoms, steps, variant
        self.base_atoms=nn.Parameter(torch.randn(2,atoms,3,3)*.05)
        self.dynamics=AtomDynamics(atoms)
        self.coeff=nn.Parameter(torch.randn(2,ch,ch,atoms)*.025)
        self.stem=ConvBlock(4,ch)
        self.tail=nn.Sequential(ConvBlock(ch,ch),nn.Conv2d(ch,1,1))

    def integrate(self, delta):
        state=self.base_atoms
        h=delta/self.steps
        t=state.new_zeros(())
        for _ in range(self.steps):
            k1=self.dynamics(state,t)
            k2=self.dynamics(state+h*k1*.5,t+h*.5)
            k3=self.dynamics(state+h*k2*.5,t+h*.5)
            k4=self.dynamics(state+h*k3,t+h)
            state=state + (h/6.)*(k1+2*k2+2*k3+k4)
            t=t+h
        return state

    def forward(self, binary, alpha, alpha_target, background=0., track='binary'):
        inp, analytic, confidence = features(binary, alpha, alpha_target, self.variant, background, track)
        delta = float(math.log(max(float(alpha),1e-6)/float(alpha_target))/3.)
        atom_set = self.integrate(self.base_atoms.new_tensor(delta))
        feat=self.stem(inp)
        for k in range(2):
            weight=torch.einsum('oim,mxy->oixy',self.coeff[k],atom_set[k])
            feat=feat + F.silu(F.conv2d(feat,weight,padding=1))
        if self.variant=='photon':
            # Proposed modification: the Bernoulli inverse + local pooling is
            # the physical base. Learn only a bounded, uncertainty-gated residual.
            # Uncertain regions receive less hallucinated high-frequency change.
            residual=.35*confidence*torch.tanh(self.tail(feat))
            return (analytic+residual).clamp(0,1)
        return torch.sigmoid(self.tail(feat))



def load_legacy(directory, device):
    """Load the original multibit-Poisson U-Nets; do not retrain/overwrite."""
    if directory is None:
        return {}
    models = {}
    for kind in ('direct', 'residual'):
        path = directory / (kind + '_best.pt')
        if not path.is_file():
            raise FileNotFoundError(
                'Missing U-Net checkpoint: %s. Use --legacy-dir for the folder containing '
                'direct_best.pt and residual_best.pt.' % path)
        ckpt = torch.load(path, map_location='cpu')  # Your own trusted checkpoints only
        model = UNetRGB(ckpt.get('channels', 24), residual=(kind == 'residual'))
        model.load_state_dict(ckpt['model'])
        models[kind] = model.to(device).eval()
        print('U-Net loaded:', path, flush=True)
    return models


def legacy_poisson_condition(rgb, alpha, background, gen):
    """Exact 3-channel conditioning used by the original Poisson-trained U-Nets.

    This is a *separately simulated multibit measurement* of the same RGB scene.
    Do not present this as the SAME 1-bit observation used by the SIC models.
    """
    y = reflectance_proxy(rgb.detach().cpu())
    count = torch.poisson(y * float(alpha) + float(background), generator=gen)
    raw = ((count - float(background)) / max(float(alpha), 1e-6)).clamp(0, 1)
    r = 3
    v = torch.arange(-r, r + 1, dtype=raw.dtype)
    k = torch.exp(-0.5 * (v / 1.)**2)
    k = k / k.sum()
    ker = (k[:, None] * k[None, :])[None, None]
    smooth = F.conv2d(F.pad(raw, (r, r, r, r), mode='reflect'), ker).clamp(0, 1)
    x = torch.cat([
        linear_to_srgb(raw),
        linear_to_srgb(smooth),
        torch.full_like(raw, math.log1p(float(alpha)) / math.log1p(40)),
    ], dim=1)
    return x, linear_to_srgb(raw).repeat(1, 3, 1, 1)

def checkpoint_path(args, name): return args.out/(name+'_last.pt')
def best_path(args,name): return args.out/(name+'_best.pt')


def save_checkpoint(args,name,model,opt,step,best,best_state=False):
    path=best_path(args,name) if best_state else checkpoint_path(args,name)
    torch.save({'model':model.state_dict(), 'optimizer':opt.state_dict() if not best_state else None,
                'step':step,'best_val':best,'name':name,
                'config':{'width':args.width,'atoms':args.atoms,'ode_steps':args.ode_steps,
                          'target_alpha':args.target_alpha,'background':args.background,
                          'size':args.size,'unet_width':args.unet_width,'track':args.track,
                          'nir_root':str(args.nir_root) if args.nir_root else None}},path)


def resume_model(args,name,model,opt):
    path=checkpoint_path(args,name)
    if args.resume and path.exists():
        ck=torch.load(path,map_location='cpu')
        config=ck.get('config',{})
        for key,val in [('width',args.width),('atoms',args.atoms),('ode_steps',args.ode_steps),
                        ('target_alpha',args.target_alpha),('background',args.background),('size',args.size),('unet_width',args.unet_width),
                        ('track',args.track),('nir_root',str(args.nir_root) if args.nir_root else None)]:
            if key in config and config[key]!=val: raise ValueError('Resume config mismatch for '+key)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        print('Resumed',name,'step',ck['step'],flush=True)
        return ck['step'],ck['best_val']
    return 0,-1e9


def endless(loader):
    while True:
        for item in loader:yield item


@torch.no_grad()
def validation(model,loader,device,args,task,variant=None):
    model.eval()
    vals=[]
    n=0
    # Fixed validation observations across checkpoints (no noisy model selection).
    g = torch.Generator().manual_seed(args.seed + (311 if task == "color" else 829))
    for batch in loader:
        rgb,y = batch_rgb_y(batch, device, args.track)
        alpha=args.target_alpha if task=='color' else args.val_alpha
        obs=observation(y.detach().cpu(),alpha,args.background,gen=g,track=args.track).to(device)
        if task=='color':
            pred=model(colorizer_input(obs,alpha,args.background,args.track));target=rgb
        elif task=='unet':
            pred=model(matched_unet_condition(obs,alpha,args.background,args.track));target=rgb
        else:
            pred=model(obs,alpha,args.target_alpha,args.background,args.track)
            target=target_expected(y,args.target_alpha,args.background,args.track)
        vals.extend(psnr_batch(pred,target).detach().cpu().tolist())
        n+=len(rgb)
        if n>=args.val_images:break
    return float(np.mean(vals))


def psnr_batch(pred,ref):
    err=(pred-ref).square().flatten(1).mean(1).clamp_min(1e-10)
    return -10.*torch.log10(err)


def train_color(args,loader,val_loader,device):
    name='colorizer'
    gen=Pix2PixColorizer(args.width).to(device)
    disc=PatchDiscriminator(max(8,args.width)).to(device) if args.gan_weight>0 else None
    opt=torch.optim.Adam(gen.parameters(),lr=args.lr,betas=(.5,.999))
    dopt=torch.optim.Adam(disc.parameters(),lr=args.lr,betas=(.5,.999)) if disc else None
    step,best=resume_model(args,name,gen,opt)
    # Save/restore discriminator explicitly for truly resumable adversarial training.
    dlast=args.out/'discriminator_last.pt'
    if disc and args.resume and dlast.exists():
        dstate=torch.load(dlast,map_location='cpu')
        disc.load_state_dict(dstate['model']);dopt.load_state_dict(dstate['optimizer'])
    iterator=endless(loader);start=time.monotonic()
    while step<args.color_steps:
        rgb,y=batch_rgb_y(next(iterator), device, args.track)
        obs=observation(y,args.target_alpha,args.background,track=args.track)
        obs=colorizer_input(obs,args.target_alpha,args.background,args.track)
        fake=gen(obs)
        if disc:
            dopt.zero_grad(set_to_none=True)
            logits_real=disc(obs,rgb)
            logits_fake=disc(obs,fake.detach())
            dloss=(F.binary_cross_entropy_with_logits(logits_real,torch.ones_like(logits_real))+
                   F.binary_cross_entropy_with_logits(logits_fake,torch.zeros_like(logits_fake)))*.5
            dloss.backward();dopt.step()
            for param in disc.parameters():param.requires_grad_(False)
        opt.zero_grad(set_to_none=True)
        fake=gen(obs)
        loss=F.l1_loss(fake,rgb)
        if disc:
            logits=disc(obs,fake)
            loss=loss + args.gan_weight*F.binary_cross_entropy_with_logits(logits,torch.ones_like(logits))
        loss.backward();opt.step()
        if disc:
            for param in disc.parameters():param.requires_grad_(True)
        step+=1
        if step%args.report_every==0 or step==args.color_steps:
            val=validation(gen,val_loader,device,args,'color')
            if val>best:
                best=val;save_checkpoint(args,name,gen,opt,step,best,True)
            save_checkpoint(args,name,gen,opt,step,best)
            if disc:
                torch.save({'model':disc.state_dict(),'optimizer':dopt.state_dict()},dlast)
            print('Colorizer %d/%d loss %.4f val %.2f best %.2f sec %.1f'%
                  (step,args.color_steps,loss.item(),val,best,time.monotonic()-start),flush=True)
            gen.train()
    return gen


def train_exposure(args,loader,val_loader,device,variant):
    name=variant
    model=AtomODE(args.width,args.atoms,args.ode_steps,variant).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    step,best=resume_model(args,name,model,opt)
    iterator=endless(loader);start=time.monotonic()
    while step<args.exposure_steps:
        rgb,y=batch_rgb_y(next(iterator), device, args.track)
        alpha=random.choice(args.train_alphas)
        obs=observation(y,alpha,args.background,track=args.track)
        target=target_expected(y,args.target_alpha,args.background,args.track)
        pred=model(obs,alpha,args.target_alpha,args.background,args.track)
        loss=F.l1_loss(pred,target)
        # photon-consistent forward likelihood surrogate: predicted reflectance
        # should agree statistically with the *observed local occupancy*.
        if variant=='photon':
            if args.track == 'binary':
                rho=(-torch.log1p(-pred.clamp(max=.995))-args.background).clamp_min(0)/args.target_alpha
                p_back=target_expected(rho,alpha,args.background,'binary')
                observed_local=F.avg_pool2d(obs,5,1,2)
            else:
                p_back=pred.clamp(0,1)*alpha+args.background
                observed_local=F.avg_pool2d(obs,5,1,2)
            loss=loss+args.phys_weight*F.smooth_l1_loss(p_back,observed_local)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();step+=1
        if step%args.report_every==0 or step==args.exposure_steps:
            val=validation(model,val_loader,device,args,'exp')
            if val>best:
                best=val;save_checkpoint(args,name,model,opt,step,best,True)
            save_checkpoint(args,name,model,opt,step,best)
            print('%s %d/%d loss %.4f val %.2f best %.2f sec %.1f'%
                  (name,step,args.exposure_steps,loss.item(),val,best,time.monotonic()-start),flush=True)
            model.train()
    return model


def train_unet(args, loader, val_loader, device, kind):
    """Train U-Net with its OWN track's observation, not legacy Poisson input."""
    name = 'unet_' + kind
    model = UNetRGB(args.unet_width, residual=(kind == 'residual')).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    step, best = resume_model(args, name, model, opt)
    iterator = endless(loader)
    start = time.monotonic()
    while step < args.unet_steps:
        rgb,y = batch_rgb_y(next(iterator), device, args.track)
        alpha = random.choice(args.train_alphas)
        obs = observation(y,alpha,args.background,track=args.track)
        cond = matched_unet_condition(obs,alpha,args.background,args.track)
        pred = model(cond)
        loss = F.l1_loss(pred,rgb) + .1*F.mse_loss(pred,rgb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
        if step % args.report_every == 0 or step == args.unet_steps:
            score = validation(model,val_loader,device,args,'unet')
            if score > best:
                best=score;save_checkpoint(args,name,model,opt,step,best,True)
            save_checkpoint(args,name,model,opt,step,best)
            print('%s %d/%d loss %.4f val %.2f best %.2f sec %.1f'%
                  (name,step,args.unet_steps,loss.item(),score,best,time.monotonic()-start),flush=True)
            model.train()
    return model


def load_model(args,name,device):
    path=best_path(args,name)
    if not path.exists():
        raise FileNotFoundError(str(path)+' missing. Run --mode train first.')
    ck=torch.load(path,map_location='cpu')
    config=ck.get('config',{})
    if abs(config.get('target_alpha',args.target_alpha)-args.target_alpha)>1e-6:
        raise ValueError('Target alpha disagrees with pretrained checkpoint')
    if config.get('track', 'binary') != args.track:
        raise ValueError('Checkpoint track differs from --track')
    if config.get('nir_root', None) != (str(args.nir_root) if args.nir_root else None):
        raise ValueError('Checkpoint NIR configuration differs from this run')
    if name=='colorizer':model=Pix2PixColorizer(config.get('width',args.width))
    elif name.startswith('unet_'):
        model=UNetRGB(config.get('unet_width',args.unet_width), residual=(name == 'unet_residual'))
    else:model=AtomODE(config.get('width',args.width),config.get('atoms',args.atoms),
                       config.get('ode_steps',args.ode_steps),name)
    model.load_state_dict(ck['model']);model.to(device).eval()
    return model


def metrics(pred,truth):
    p=np.clip(pred[0].detach().cpu().permute(1,2,0).numpy(),0,1)
    t=truth[0].detach().cpu().permute(1,2,0).numpy()
    psnr=-10*math.log10(max(float(np.mean((p-t)**2)),1e-10))
    v={'psnr':psnr,'ssim':float('nan'),'delta_e76':float('nan'),'ab_error':float('nan')}
    if sk_ssim is not None:
        v['ssim']=float(sk_ssim(t,p,data_range=1,channel_axis=-1))
    if rgb2lab is not None:
        lab=rgb2lab(p);target=rgb2lab(t)
        v['delta_e76']=float(np.sqrt(((lab-target)**2).sum(2)).mean())
        v['ab_error']=float(np.sqrt(((lab[:,:,1:]-target[:,:,1:])**2).sum(2)).mean())
    return v


def to_img(t):
    a=t.detach().cpu()
    if a.ndim==4:a=a[0]
    if a.shape[0]==1:a=a.repeat(3,1,1)
    a=np.uint8(a.permute(1,2,0).clamp(0,1).numpy()*255+.5)
    return Image.fromarray(a)


def make_gallery(rows,names,size,path):
    # Upscale small 64x64 pilot results for readable side-by-side headings.
    cell=max(128,size)
    left=164;header=36;rh=cell+4
    image=Image.new('RGB',(left+cell*len(names),header+rh*len(rows)),(255,255,255))
    draw=ImageDraw.Draw(image)
    for j,name in enumerate(names):draw.text((left+cell*j+4,9),name,fill=(0,0,0))
    for i,(label,images) in enumerate(rows):
        y=header+i*rh
        draw.text((6,y+8),label,fill=(0,0,0))
        for j,im in enumerate(images):
            if im.size!=(cell,cell):im=im.resize((cell,cell),Image.Resampling.BICUBIC)
            image.paste(im,(left+cell*j,y))
    image.save(path)


@torch.no_grad()
def evaluate(args, files, device):
    print('Evaluating held-out split, track=%s...' % args.track,flush=True)
    keys=('colorizer','atomode','photon','unet_direct','unet_residual')
    available={k:best_path(args,k).exists() for k in keys}
    if not any(available.values()):
        raise FileNotFoundError('No checkpoints in %s; train first' % args.out)
    models={k:load_model(args,k,device) for k in keys if available[k]}
    legacy=load_legacy(args.legacy_dir,device) if args.legacy_dir else {}
    # Each row uses ONE measurement type for the trained models.
    names=['GT RGB', '1-bit SPAD' if args.track=='binary' else 'Active Poisson counts',
           'Analytic exposure']
    if 'unet_direct' in models: names.append('Direct U-Net')
    if 'unet_residual' in models: names.append('Residual U-Net')
    if 'colorizer' in models:
        names.extend(['Pix2Pix direct','Analytic + Pix2Pix'])
        if 'atomode' in models:names.append('AtomODE SIC')
        if 'photon' in models:names.append('PhotonODE SIC')
    if legacy:
        names.extend(['Legacy Poisson input','Legacy Direct (OOD)','Legacy Residual (OOD)'])
    test_paths=files['test'][:args.eval_images] if args.eval_images>0 else files['test']
    dataset=RGBDataset(test_paths,args.size,False,args.nir_root,
                       args.data_root/'imagenette2-160')
    rows, results=[],[]
    for idx in range(len(dataset)):
        sample=dataset[idx]
        rgb,y=batch_rgb_y((sample[0][None],sample[1][None]) if isinstance(sample,tuple) else sample[None],
                          device,args.track)
        path=test_paths[idx]
        photo=to_img(rgb)
        for alpha in args.test_alphas:
            rng=torch.Generator().manual_seed(args.seed + 10000*idx + round(alpha*100))
            obs=observation(y.detach().cpu(),alpha,args.background,gen=rng,track=args.track).to(device)
            _,analytic,_=features(obs,alpha,args.target_alpha,'photon',args.background,args.track)
            display={'GT RGB':photo, names[1]:to_img(observation_display(obs,alpha,args.background,args.track)),
                     'Analytic exposure':to_img(analytic)}
            preds={}
            unet_x=matched_unet_condition(obs,alpha,args.background,args.track)
            if 'unet_direct' in models:preds['Direct U-Net']=models['unet_direct'](unet_x)
            if 'unet_residual' in models:preds['Residual U-Net']=models['unet_residual'](unet_x)
            if 'colorizer' in models:
                color=models['colorizer']
                preds['Pix2Pix direct']=color(colorizer_input(obs,alpha,args.background,args.track))
                preds['Analytic + Pix2Pix']=color(analytic)
                if 'atomode' in models:
                    exposure=models['atomode'](obs,alpha,args.target_alpha,args.background,args.track)
                    preds['AtomODE SIC']=color(exposure)
                if 'photon' in models:
                    exposure=models['photon'](obs,alpha,args.target_alpha,args.background,args.track)
                    preds['PhotonODE SIC']=color(exposure)
            if legacy:
                lg_rng=torch.Generator().manual_seed(args.seed+10000*idx+round(alpha*100)+500000)
                x_old,visible=legacy_poisson_condition(rgb,alpha,args.unet_background,lg_rng)
                display['Legacy Poisson input']=to_img(visible)
                preds['Legacy Direct (OOD)']=legacy['direct'](x_old.to(device))
                preds['Legacy Residual (OOD)']=legacy['residual'](x_old.to(device))
            for label,pred in preds.items():
                display[label]=to_img(pred)
                results.append(dict(file=str(path),track=args.track,alpha=alpha,method=label,
                                    **metrics(pred,rgb)))
            if idx<args.gallery_images:
                rows.append(('%s\nalpha=%g'%(path.stem[:19],alpha),[display[k] for k in names]))
        print('Tested %d/%d %s'%(idx+1,len(test_paths),path.name),flush=True)
    with (args.out/'test_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['file','track','alpha','method','psnr','ssim','delta_e76','ab_error'])
        w.writeheader();w.writerows(results)
    stats=[]
    for alpha in args.test_alphas:
        for name in names:
            subset=[r for r in results if r['alpha']==alpha and r['method']==name]
            if not subset:continue
            d={'track':args.track,'alpha':alpha,'method':name,'n':len(subset)}
            for col in ('psnr','ssim','delta_e76','ab_error'):
                d[col]=round(float(np.nanmean([r[col] for r in subset])),4)
            stats.append(d)
    with (args.out/'summary_by_alpha.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['track','alpha','method','n','psnr','ssim','delta_e76','ab_error'])
        w.writeheader();w.writerows(stats)
    if rows:make_gallery(rows,names,args.size,args.out/'comparison_gallery.png')
    print('Saved',args.out.resolve(),flush=True)

def main():
    ap=argparse.ArgumentParser(description='SIC integrated into original Mac SPAD quickbench pipeline')
    ap.add_argument('--mode',choices=['train','eval'],default='train')
    ap.add_argument('--track',choices=['binary','active'],default='binary',
                    help='binary=passive single-bit; active=laser photon-count imaging')
    ap.add_argument('--nir-root',type=Path,default=None,
                    help='OPTIONAL paired NIR images under train/... and val/... mirroring RGB dataset')
    ap.add_argument('--data-root',type=Path,default=Path('data'))
    ap.add_argument('--no-download', action='store_true')
    ap.add_argument('--out',type=Path,default=Path('runs/atomode_sic'))
    ap.add_argument('--legacy-dir',type=Path,default=None)
    ap.add_argument('--train-count',type=int,default=800)
    ap.add_argument('--val-count',type=int,default=100)
    ap.add_argument('--test-count',type=int,default=100)
    ap.add_argument('--gallery-images',type=int,default=6)
    ap.add_argument('--eval-images',type=int,default=6,help='Number of held-out images to evaluate (0 = entire test split)')
    ap.add_argument('--unet-background',type=float,default=.2,help='Background counts/pixel for original U-Net Poisson simulator')
    ap.add_argument('--val-images',type=int,default=16)
    ap.add_argument('--size',type=int,default=128)
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--width',type=int,default=16)
    ap.add_argument('--atoms',type=int,default=6)
    ap.add_argument('--ode-steps',type=int,default=2)
    ap.add_argument('--gan-weight',type=float,default=.02)
    ap.add_argument('--phys-weight',type=float,default=.12)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--color-steps',type=int,default=1000)
    ap.add_argument('--unet-steps',type=int,default=1000)
    ap.add_argument('--unet-width',type=int,default=24)
    ap.add_argument('--exposure-steps',type=int,default=800)
    ap.add_argument('--report-every',type=int,default=100)
    ap.add_argument('--train-alphas',type=float,nargs='+',default=[.5,1.,3.,6.,10.])
    ap.add_argument('--test-alphas',type=float,nargs='+',default=[1.,3.,10.])
    ap.add_argument('--val-alpha',type=float,default=1.)
    ap.add_argument('--target-alpha',type=float,default=3.)
    ap.add_argument('--background',type=float,default=.05)
    ap.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--train-module',choices=['all','colorizer','atomode','photon','unets','unet_direct','unet_residual'],default='all')
    ap.add_argument('--skip-eval',action='store_true')
    args=ap.parse_args()
    if args.size%4:raise ValueError('--size must be divisible by 4')
    if args.target_alpha<=0:raise ValueError('--target-alpha must be positive')
    if args.nir_root is not None and args.track!='active':
        raise ValueError('--nir-root only applies to --track active')
    if args.nir_root is None and args.track=='active':
        print('WARNING: ACTIVE TRACK USES RGB-LUMINANCE PROXY, NOT REAL NIR REFLECTANCE',flush=True)
    if any(a<=0 for a in args.train_alphas+args.test_alphas):raise ValueError('alphas must be >0')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    seed_all(args.seed)
    device=device_auto(args.device)
    print('Device',device,'PyTorch',torch.__version__,flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='train' and not args.resume:
        names=['colorizer','atomode','photon','unet_direct','unet_residual'] if args.train_module=='all' else (
            ['unet_direct','unet_residual'] if args.train_module=='unets' else [args.train_module])
        conflicts=[str(checkpoint_path(args,k)) for k in names if checkpoint_path(args,k).exists()]
        if conflicts:
            raise FileExistsError('Existing training checkpoints would be overwritten: '+', '.join(conflicts)+
                                  '; use --resume or a different --out.')
    track_file=args.out/'measurement_track.json'
    track_cfg={'track':args.track, 'nir_root':str(args.nir_root) if args.nir_root else None}
    if track_file.exists():
        if json.loads(track_file.read_text())!=track_cfg:
            raise ValueError('This --out was used with another measurement track/NIR source; use a separate --out')
    elif args.mode=='eval' and any(args.out.glob('*_best.pt')):
        raise ValueError('Existing checkpoints have no track metadata; use a new --out for the two-track experiment')
    elif args.mode=='train':
        track_file.write_text(json.dumps(track_cfg,indent=2))
    files=split_files(args)
    print('Scene split:',{k:len(v) for k,v in files.items()},flush=True)
    if args.mode=='train':
        base=args.data_root/'imagenette2-160'
        loader=DataLoader(RGBDataset(files['train'],args.size,True,args.nir_root,base),
                          batch_size=args.batch,shuffle=True,num_workers=0)
        val=DataLoader(RGBDataset(files['val'],args.size,False,args.nir_root,base),
                       batch_size=1,num_workers=0)
        if args.train_module in ('all','colorizer'):
            train_color(args,loader,val,device)
        if args.train_module in ('all','atomode'):
            train_exposure(args,loader,val,device,'atomode')
        if args.train_module in ('all','photon'):
            train_exposure(args,loader,val,device,'photon')
        if args.train_module in ('all','unets','unet_direct'):
            train_unet(args,loader,val,device,'direct')
        if args.train_module in ('all','unets','unet_residual'):
            train_unet(args,loader,val,device,'residual')
    if not args.skip_eval:
        evaluate(args,files,device)


if __name__=='__main__':
    main()
