#!/usr/bin/env python3
"""Consolidated V15/V17/V18 inference components. Faithful state_dict compatibility.

Combined from V15/V16/V17 compatibility snapshots and V18, no custom plugin modules.
No ground truth is accessed by non-oracle predictions.
"""
import argparse, csv, math, random, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

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


def features_from_counts(k, budget, args, variant=None):
    """Five V15 photon features from measured aggregate hit counts (NO RGB GT).

    k shape: [N,1,H,W], float counts; never resize a count image.
    Only valid for a calibrated static scene / same acquisition model as training.
    """
    if k.ndim != 4 or k.shape[1] != 1:
        raise ValueError('Expected aggregate counts [N,1,H,W]')
    if not torch.isfinite(k).all() or (k < 0).any():
        raise ValueError('Counts must be nonnegative and finite')
    if args.mode == 'bernoulli':
        if (k > args.cycles).any():
            raise ValueError('Bernoulli counts cannot exceed --cycles')
        a, b = budget/args.cycles, args.bg/args.cycles
        m = (k.mean(dim=(-2,-1),keepdim=True)/args.cycles).clamp(1e-4,1-1e-4)
        phat = ((k + m)/(args.cycles+1.)).clamp(1e-7,1-1e-7)
        est = (-torch.log1p(-phat)-b).clamp_min(0)/max(a,1e-9)
        var = phat / ((args.cycles*a*a)*(1-phat)).clamp_min(1e-12)
    elif args.mode == 'poisson':
        est = ((k-args.bg)/max(budget,1e-9)).clamp_min(0)
        var = (k+0.5)/max(budget*budget,1e-12)
    else: raise ValueError(args.mode)
    bf = precision_bilateral(est, var,args.bf_radius,args.bf_sigma,args.bf_knoise,args.bf_prec_cap)
    raw=(torch.log1p(k)/math.log1p(max(2,args.cycles))).clamp(0,1)
    unc=(torch.log1p(var)/math.log1p(100.)).clamp(0,1)
    exposure=torch.full_like(raw,math.log1p(budget)/math.log1p(32.))
    feats=torch.cat([raw,est.clamp(0,1),bf.clamp(0,1),unc,exposure],dim=1)
    if variant is not None: feats=make_variant(feats,variant)
    return feats,k,est,var,bf


def make_features(rgb,budget,args,variant=None):
    k,_,_=observe(rgb,budget,args.bg,args.cycles,args.mode)
    return features_from_counts(k,budget,args,variant)


def photon_nll(y,k,budget,args):
    """Aggregated Bernoulli/Poisson NLL, normalized per expected photon budget."""
    if args.mode == 'bernoulli':
        p=(-torch.expm1(-(budget*y+args.bg)/args.cycles)).clamp(1e-7,1-1e-7)
        nll=-(k*torch.log(p)+(args.cycles-k)*torch.log1p(-p))
    else:
        mu=(budget*y+args.bg).clamp_min(1e-8)
        nll=mu-k*torch.log(mu)
    return nll.mean()/max(budget,.1)


def edge_l1(pred,target):
    return (pred[...,1:,:]-pred[...,:-1,:]-target[...,1:,:]+target[...,:-1,:]).abs().mean() + \
           (pred[...,:,1:]-pred[...,:,:-1]-target[...,:,1:]+target[...,:,:-1]).abs().mean()


def rgb_training_loss(pred,gt,k,budget,args):
    rgb=F.l1_loss(pred,gt); ypred=luminance(pred); ygt=luminance(gt)
    yl=F.l1_loss(ypred,ygt); ph=photon_nll(ypred,k,budget,args)
    return rgb+args.lambda_y*yl+args.lambda_ph*ph, (rgb,yl,ph)


def adapter_training_loss(pred,gt,k,budget,args):
    yt=luminance(gt); yl=F.l1_loss(pred,yt)
    edge=edge_l1(pred,yt); ph=photon_nll(pred,k,budget,args)
    return yl+args.lambda_edge*edge+args.lambda_ph*ph,(yl,edge,ph)


def bridge_flow_loss(net,z0,z1,features,budget,noise_sigma):
    b=z0.shape[0]
    t=torch.rand((b,),device=z0.device).clamp(.001,.999)
    ts=t[:,None,None,None]
    eps=torch.randn_like(z0)
    zt=(1-ts)*z0+ts*z1+noise_sigma*torch.sin(math.pi*ts)*eps
    vt=z1-z0+noise_sigma*math.pi*torch.cos(math.pi*ts)*eps
    pred=net(zt,z0,features,t,budget)
    return (pred-vt).square().mean(),zt,pred,t,eps


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


def seed_photons(args, index, budget, salt=0):
    # Seeds independent of variant, same underlying observations for comparisons.
    torch.manual_seed(args.seed + 1_100_003 + index*7919 + int(round(budget*10000)) + salt)


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
    gray_srgb = linear_to_srgb(y_linear.clamp(0, 1))[0, 0].cpu().numpy()
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
    return srgb_to_linear(p).clamp(0, 1)


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
    net = RGBUNet(ck['base']).to(device)
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
    ps = linear_to_srgb(pred)[0].permute(1,2,0).cpu().numpy().astype(np.float32)
    gs = linear_to_srgb(gt)[0].permute(1,2,0).cpu().numpy().astype(np.float32)
    lab_p = cv2.cvtColor(np.ascontiguousarray(ps), cv2.COLOR_RGB2LAB)
    lab_g = cv2.cvtColor(np.ascontiguousarray(gs), cv2.COLOR_RGB2LAB)
    delta = lab_p-lab_g
    ab = np.sqrt(np.sum(delta[...,1:]**2, axis=2)).mean()
    return {
        'psnr_srgb': float(-10*math.log10(max(np.mean((ps-gs)**2),1e-12))),
        'ssim_srgb': float(ssim_srgb(ps,gs)),
        'lab_ab_error': float(ab),
        'delta_e76': float(np.sqrt(np.sum(delta**2,axis=-1)).mean()),
        'linear_luma_mse': float(((luminance(pred)-luminance(gt))**2).mean()),
        'linear_rgb_L1': float((pred-gt).abs().mean()),
    }


def to_pil_srgb(img_linear):
    p = linear_to_srgb(img_linear)
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


class LumaAdapter(nn.Module):
    """5-channel photon features -> one linear-light luminance channel in [0,1]."""
    def __init__(self, base=32):
        super().__init__()
        self.e1 = ResidualBlock(5, base)
        self.e2 = ResidualBlock(base, base*2)
        self.mid = ResidualBlock(base*2, base*4)
        self.d2 = ResidualBlock(base*6, base*2)
        self.d1 = ResidualBlock(base*3, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.mid(F.avg_pool2d(b, 2))
        d = self.d2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], dim=1))
        e = self.d1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], dim=1))
        return torch.sigmoid(self.out(e))


def read_vae(args, device):
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise RuntimeError('Install: python -m pip install diffusers transformers accelerate safetensors') from exc
    print('Loading pretrained frozen VAE:',args.vae_model,flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device).eval()
    vae.requires_grad_(False)
    factor = float(getattr(vae.config, 'scaling_factor', 0.18215))
    return vae, factor


def vae_encode(vae, s_rgb, factor):
    """s_rgb is gamma-encoded sRGB in [0,1]; deterministic VAE posterior mode."""
    return vae.encode(s_rgb.mul(2).sub(1)).latent_dist.mode() * factor


def vae_decode(vae, z, factor):
    # Freeze decoder parameters but permit gradients to its input z during training.
    return vae.decode(z / factor).sample.add(1).div(2).clamp(0,1)


def y_to_srgb3(y):
    return linear_to_srgb(y.clamp(0,1)).repeat(1,3,1,1)


class Res(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        g = 8 if cout % 8 == 0 else 1
        self.layers = nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(g,cout),nn.SiLU(),
                                    nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(g,cout),nn.SiLU())
        self.skip = nn.Conv2d(cin,cout,1)
    def forward(self,x):return self.layers(x)+self.skip(x)


class Bridge(nn.Module):
    """Time-conditioned latent vector field v(z_t,z_source,photon_features,t,budget).

    VAE: 4 channels at 1/8 resolution. Observation channels: five V15 features.
    U-Net is purposely small to enable a scientifically useful first ablation.
    """
    def __init__(self, base=64, latent_channels=4, feat_channels=5):
        super().__init__()
        self.latent_channels=latent_channels
        cin=latent_channels*2 + feat_channels + 2  # z_t, z0, features, t, budget
        self.a=Res(cin,base);self.b=Res(base,base*2);self.c=Res(base*2,base*4)
        self.d=Res(base*6,base*2);self.e=Res(base*3,base)
        self.out=nn.Conv2d(base,latent_channels,3,padding=1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,z_t,z0,features,t,budget):
        shape=z_t.shape[-2:]
        c=F.interpolate(features,size=shape,mode='area')
        bs=z_t.size(0)
        if not torch.is_tensor(t): t=torch.full((bs,),float(t),device=z_t.device,dtype=z_t.dtype)
        else: t=t.reshape(bs)
        tb=t[:,None,None,None].expand(bs,1,*shape)
        bb=torch.full_like(tb,math.log1p(float(budget))/math.log(9.))
        x=torch.cat([z_t,z0,c,tb,bb],1)
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));m=self.c(F.avg_pool2d(b,2))
        d=self.d(torch.cat((F.interpolate(m,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        e=self.e(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return self.out(e)


def bridge_sample(net,z0,feats,budget,steps=8,stochastic_start=0.0):
    if steps < 1: raise ValueError('--steps must be >=1')
    z=z0+stochastic_start*torch.randn_like(z0)
    # Midpoint integration (2 network evaluations per step) significantly more
    # stable than one large Euler step for conditional flows.
    dt=1.0/steps
    for j in range(steps):
        t=(j+.0)*dt
        v1=net(z,z0,feats,t,budget)
        v2=net(z+dt*.5*v1,z0,feats,t+dt*.5,budget)
        z=z+dt*v2
    return z


def fit_luminance(rgb_srgb,ylin,weight):
    """Post-hoc luminance adjustment, preserving approximate RGB chroma differences.

    Method is intentionally a separate ablation; clipping may affect accuracy.
    """
    if weight<=0:return rgb_srgb
    lin=srgb_to_linear(rgb_srgb.clamp(0,1))
    delta=(ylin-luminance(lin))*weight
    lin=(lin+delta).clamp(0,1)
    return linear_to_srgb(lin)




def load_adapter(args, device):
    """Load original V17 checkpoint without importing the historical trainer."""
    path = args.v17_out / 'luma_adapter_best.pt'
    if not path.is_file():
        raise FileNotFoundError(f'V17 checkpoint not found: {path}')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('mode','cycles','bg','size','bf_radius','bf_sigma','bf_knoise',
                'bf_prec_cap','seed','val_count'):
        if ck.get(key) != getattr(args, key):
            raise ValueError(f'V17 checkpoint mismatch: {key}, ck={ck.get(key)}, cli={getattr(args,key)}')
    net = LumaAdapter(ck['base']).to(device)
    net.load_state_dict(ck['state_dict'], strict=True)
    net.eval()
    return net


def load_bridge(args, device):
    path = args.v18_out / 'bridge_best.pt'
    if not path.is_file():
        raise FileNotFoundError(f'V18 checkpoint not found: {path}')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    for key in ('mode','cycles','bg','size','seed','val_count','vae_model'):
        if cfg[key] != getattr(args,key):
            raise ValueError(f'V18 checkpoint mismatch: {key}: {cfg[key]} != {getattr(args,key)}')
    net = Bridge(cfg['base'],cfg['latent_channels']).to(device)
    net.load_state_dict(ck['model'], strict=True)
    net.eval()
    return net


@torch.inference_mode()
def bridge_predict(vae, factor, net, y, features, budget, steps, luma_fusion):
    z0 = vae_encode(vae, y_to_srgb3(y), factor)
    z = bridge_sample(net, z0, features, budget, steps=steps)
    rgb = vae_decode(vae, z, factor)
    if luma_fusion > 0:
        rgb = fit_luminance(rgb,y,luma_fusion)
    return srgb_to_linear(rgb)


def get_device(setting):
    name = 'cuda' if setting=='auto' and torch.cuda.is_available() else ('cpu' if setting=='auto' else setting)
    if name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    return torch.device(name)
