#!/usr/bin/env python3
"""No-training DPIR DRUNet / exact 64-binomial PnP benchmark for SPAD single-channel Y.
Requires existing v17_v15_compat, v17_v16_compat, train_eval_v17 and V15,V17 ckpts.
Official DPIR architecture and publicly hosted weights are acquired on the first run.
All hyperparameters are selected on a disjoint Imagenette validation subset before test/VOC.
No GT data enters the reconstruction function. Do not confuse GT-based metric/Oracle with deployment.
"""
import argparse, csv, hashlib, json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import v17_v15_compat as v15
import v17_v16_compat as v16
import train_eval_v17 as v17

# Pre-registered small candidate set; change only on validation (not on test).
PRESETS = {'gentle':(4,12.,10./255.), 'strong':(4,40.,20./255.)}


def write_csv(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def device_of(args):
    if args.device=='auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    return torch.device(args.device)


def _download_file(url,path,min_bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    req=urllib.request.Request(url,headers={'User-Agent':'SPAD-DPIR-experiment/1.0'})
    print('Downloading',url,flush=True)
    with urllib.request.urlopen(req,timeout=60) as response,tmp.open('wb') as target:
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            target.write(chunk)
    if tmp.stat().st_size<min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError('Incomplete download; size less than expected for '+str(path))
    tmp.replace(path)


def ensure_dpir_model(args,device):
    root=args.dpir_root
    mods=root/'models'
    for name in ('network_unet.py','basicblock.py'):
        f=mods/name
        if not f.is_file():
            _download_file(f'https://raw.githubusercontent.com/cszn/DPIR/master/models/{name}',f,5000)
    (mods/'__init__.py').touch(exist_ok=True)
    model_file=args.weights if args.weights else root/'model_zoo'/'drunet_gray.pth'
    if not model_file.is_file():
        model_file.parent.mkdir(parents=True,exist_ok=True)
        # HF download is usually more reliable on Windows; verify published 131MB SHA256.
        try:
            from huggingface_hub import hf_hub_download
            print('Downloading public DRUNet gray checkpoint via Hugging Face (first run only)',flush=True)
            src=Path(hf_hub_download('deepinv/drunet',filename='drunet_gray.pth'))
            import shutil
            shutil.copy2(src,model_file)
        except Exception as exc:
            print('HuggingFace download unavailable:',str(exc)[:250],flush=True)
            _download_file('https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth',model_file,10000000)
    if model_file.stat().st_size<10000000:raise RuntimeError('DRUNet checkpoint suspiciously small: '+str(model_file))
    # A deterministic architecture import; DPIR defines 'models.basicblock' as absolute import.
    sys.path.insert(0,str(root.resolve()))
    from models.network_unet import UNetRes
    net=UNetRes(in_nc=2,out_nc=1,nc=[64,128,256,512],nb=4,act_mode='R',
                downsample_mode='strideconv',upsample_mode='convtranspose')
    try: weights=torch.load(model_file,map_location='cpu',weights_only=True)
    except TypeError: weights=torch.load(model_file,map_location='cpu')
    if isinstance(weights,dict) and 'state_dict' in weights:weights=weights['state_dict']
    net.load_state_dict(weights,strict=True)
    net.to(device).eval().requires_grad_(False)
    print('Loaded official DRUNet architecture / grayscale weights:',model_file,flush=True)
    return net


def exact_binomial_prox_torch(z,k,B,T,bg,rho,iterations=32):
    """Convex per-pixel data-fidelity proximal on linear Y, tested against SciPy/Numpy version."""
    if B<=0 or T<=0 or bg<0 or rho<=0:raise ValueError('Invalid acquisition/proximal parameters')
    z=z.float();k=k.float();lo=torch.zeros_like(z);hi=torch.ones_like(z)
    def d(y):
        p=(-torch.expm1(-(B*y+bg)/T)).clamp_min(1e-12)
        return (B/T)*(T-k/p)+rho*(y-z)
    at_lo=d(lo)>=0;at_hi=d(hi)<=0
    for _ in range(iterations):
        m=.5*(lo+hi)
        below=d(m)<0
        lo=torch.where(below,m,lo);hi=torch.where(below,hi,m)
    return torch.where(at_lo,torch.zeros_like(z),torch.where(at_hi,torch.ones_like(z),.5*(lo+hi)))


@torch.inference_mode()
def drunet_denoise(net,y,sigma):
    """DRUNet learned on display-domain gray images; transform linear Y -> neutral sRGB."""
    display=v15.linear_to_srgb(y.clamp(0,1))
    net_input=torch.cat([display,torch.full_like(display,float(sigma))],dim=1)
    # 128 px inputs are divisible by 8 as required by the official UNetRes.
    h,w=display.shape[-2:];pad_h=(-h)%8;pad_w=(-w)%8
    if pad_h or pad_w:net_input=F.pad(net_input,(0,pad_w,0,pad_h),mode='replicate')
    pred=net(net_input)[...,:h,:w].clamp(0,1)
    return v15.srgb_to_linear(pred).clamp(0,1)


@torch.inference_mode()
def run_pnp(net,initial,k,B,args,preset):
    iters,rho,sigma=PRESETS[preset]
    z=initial.float().clamp(0,1)
    for _ in range(iters):
        x=exact_binomial_prox_torch(z,k,B,args.cycles,args.bg,rho)
        z=drunet_denoise(net,x,sigma)
    # Final data step: output is tied to observed Bernoulli K, not pure denoiser output.
    return exact_binomial_prox_torch(z,k,B,args.cycles,args.bg,rho)


def highpass_error(a,b):
    # Test finite differences, no GT-fed reconstruction.
    dx=(a[...,1:]-a[...,:-1])-(b[...,1:]-b[...,:-1])
    dy=(a[...,1:,:]-a[...,:-1,:])-(b[...,1:,:]-b[...,:-1,:])
    return float(.5*(dx.abs().mean()+dy.abs().mean()))


def y_metrics(pred,ygt):
    return {'luma_mae':float((pred-ygt).abs().mean()),
            'luma_mse':float((pred-ygt).square().mean()),
            'edge_mae':highpass_error(pred,ygt)}


def nll_per_cycle(y,k,B,T,bg):
    p=(-torch.expm1(-(B*y+bg)/T)).clamp(1e-7,1-1e-7)
    return float((-(k*torch.log(p)+(T-k)*torch.log1p(-p))/T).mean())


def make_models(args,device):
    # v17 loader accepts original args with a relocated 'out' pointing to V17.
    import argparse
    a=argparse.Namespace(**vars(args));a.out=args.v17_out
    adapter=v17.load_adapter(a,device).eval().requires_grad_(False)
    drunet=ensure_dpir_model(args,device)
    return adapter,drunet


@torch.inference_mode()
def image_measures(gt,idx,B,args,adapter,drunet,chosen=None):
    v15.seed_photons(args,idx,B,salt=300000)
    f,k,est,var,bf=v15.make_features(gt,B,args)
    y0=adapter(f).clamp(0,1)
    direct=drunet_denoise(drunet,y0,args.direct_sigma)
    outs={'adapter':y0,'drunet_direct':direct}
    if chosen is None:
        for pr in PRESETS:outs['pnp_'+pr]=run_pnp(drunet,y0,k,B,args,pr)
    else:outs['pnp']=run_pnp(drunet,y0,k,B,args,chosen)
    return outs,k,bf


def load_gt_imagenette(args,split,idx,device):
    ds=v15.RGBImages(args.data,split,args.size,args.seed,args.val_count)
    gt,path=ds[idx]
    return gt[None].to(device),path


@torch.inference_mode()
def do_val(args,device,adapter,drunet):
    ds=v15.RGBImages(args.data,'val',args.size,args.seed,args.val_count)
    n=min(len(ds),args.val_images)
    print('Selecting PnP parameters on',n,'Imagenette validation images',flush=True)
    rows=[]
    for i in range(n):
        gt,src=ds[i];gt=gt[None].to(device);yg=v15.luminance(gt)
        for B in args.budgets:
            outs,k,bf=image_measures(gt,i,B,args,adapter,drunet)
            for m,y in outs.items():
                rows.append({'scene':src,'index':i,'budget_total':B,'method':m,**y_metrics(y,yg),
                             'measurement_nll':nll_per_cycle(y,k,B,args.cycles,args.bg)})
        if (i+1)%5==0 or i+1==n:print('Validation',i+1,'/',n,flush=True)
    write_csv(args.out/'validation'/'per_image.csv',rows)
    summary=[];chosen={}
    for B in args.budgets:
        for m in ['adapter','drunet_direct']+['pnp_'+k for k in PRESETS]:
            group=[r for r in rows if r['budget_total']==B and r['method']==m]
            summary.append({'budget_total':B,'method':m,'images':len(group),
                            **{col:float(np.mean([r[col] for r in group])) for col in ('luma_mae','luma_mse','edge_mae','measurement_nll')}})
        # PREDECLARED scoring metric based on both reconstruction fidelity and structure;
        # no test/VOC labels are accessed in model selection.
        pnp_rows=[r for r in summary if r['budget_total']==B and r['method'].startswith('pnp_')]
        best=min(pnp_rows,key=lambda r: r['luma_mae']+.5*r['edge_mae'])
        chosen[str(B)]=best['method'].removeprefix('pnp_')
        baseline=next(r for r in summary if r['budget_total']==B and r['method']=='adapter')
        delta=best['luma_mae']-baseline['luma_mae']
        print(f'B={B:g}, PnP best={chosen[str(B)]}, Y MAE change vs adapter={delta:+.5f}',flush=True)
    write_csv(args.out/'validation'/'summary.csv',summary)
    (args.out/'selected.json').write_text(json.dumps({'selected':chosen,'presets':PRESETS,
                'criterion':'val mean linear Y MAE + 0.5 * val edge MAE',
                'val_n':n,'budgets':args.budgets},indent=2),encoding='utf-8')
    return chosen


def to_gray_pil(y):
    s=v15.linear_to_srgb(y)[0,0].detach().cpu().numpy()
    return Image.fromarray(np.rint(s.clip(0,1)*255).astype(np.uint8),'L').convert('RGB')


def make_gallery(path,pairs,size):
    v16.save_gallery(pairs,size,path)


def load_colorizer(args,device):
    return v16.load_ddcolor(args,device)[0]



def optional_v20_edge(args,device):
    ckfile=args.v20_out/'models'/'edge'/'best.pt'
    if not ckfile.is_file():
        print('V20 Edge checkpoint absent; skipping optional comparator:',ckfile,flush=True)
        return None
    import train_eval_v20 as v20
    ck=torch.load(ckfile,map_location='cpu',weights_only=False)
    for key in ('mode','cycles','bg','size','seed','val_count','bf_radius','bf_sigma','bf_knoise','bf_prec_cap'):
        if ck['acquisition'][key]!=getattr(args,key):
            raise ValueError(f'V20 mismatch: {key}. Keep existing acquisition settings unchanged')
    net=v20.StructuralResidual(ck['base'],ck['max_delta']).to(device).eval().requires_grad_(False)
    net.load_state_dict(ck['model'],strict=True)
    print('Loaded optional V20 Edge comparator:',ckfile,flush=True)
    return net

@torch.inference_mode()
def do_test(args,device,adapter,drunet,chosen):
    ds=v15.RGBImages(args.data,'test',args.size,args.seed,args.val_count)
    n=min(len(ds),args.test_images) if args.test_images else len(ds)
    baseline=v16.load_v15(args,device,'full')
    v20edge=optional_v20_edge(args,device)
    colorizer=load_colorizer(args,device)
    out=args.out/'imagenette_test';out.mkdir(parents=True,exist_ok=True)
    lr=[];rgb=[]
    for idx in range(n):
        gt,src=ds[idx];gt=gt[None].to(device);yg=v15.luminance(gt)
        gallery=[]
        for B in args.budgets:
            youts,k,bf=image_measures(gt,idx,B,args,adapter,drunet,chosen[str(B)])
            v15.seed_photons(args,idx,B,salt=300000)
            feat,*_=v15.make_features(gt,B,args)
            full=baseline(v16.make_variant(feat,'full'))
            outputs={'v15_full':full}
            if v20edge is not None:youts['v20_edge']=v20edge(feat,youts['adapter']).clamp(0,1)
            for m,y in youts.items():
                lr.append({'image_index':idx,'scene':src,'budget_total':B,'method':m,
                           **y_metrics(y,yg),'measurement_nll':nll_per_cycle(y,k,B,args.cycles,args.bg)})
                outputs[m+'_ddcolor']=v16.ddcolorize(colorizer,y,device)
            for m,im in outputs.items():
                met=v16.compute_metrics(im,gt)
                rgb.append({'image_index':idx,'scene':src,'budget_total':B,'method':m,
                            **met,'measurement_nll':nll_per_cycle(v15.luminance(im),k,B,args.cycles,args.bg)})
            if idx<args.gallery_n:
                cols=[('RGB GT',v16.to_pil_srgb(gt)),('Counts',to_gray_pil((k/3).clamp(0,1))),
                      ('V17 Y',to_gray_pil(youts['adapter'])),('DRUNet Y',to_gray_pil(youts['drunet_direct'])),
                      ('PnP Y',to_gray_pil(youts['pnp']))]
                cols.extend((m,v16.to_pil_srgb(im)) for m,im in outputs.items())
                gallery.append((B,cols))
        if gallery:make_gallery(out/f'{idx:04d}.png',gallery,args.size)
        if (idx+1)%5==0 or idx+1==n:print('Imagenette test',idx+1,'/',n,flush=True)
    write_csv(out/'luma_per_image.csv',lr);write_csv(out/'rgb_per_image.csv',rgb)
    for records,filename,keys in [(lr,'luma_summary.csv',['luma_mae','luma_mse','edge_mae','measurement_nll']),
                                  (rgb,'rgb_summary.csv',['psnr_srgb','ssim_srgb','lab_ab_error','delta_e76','linear_luma_mse','measurement_nll'])]:
        summ=[]
        for B in args.budgets:
            for m in sorted(set(r['method'] for r in records)):
                r=[row for row in records if row['budget_total']==B and row['method']==m]
                if r:summ.append({'budget_total':B,'method':m,'images':len(r),
                          **{k:float(np.mean([v[k] for v in r])) for k in keys}})
        write_csv(out/filename,summ)
    print('Saved Imagenette test:',out,flush=True)


def voc_palette():
    tab=np.zeros((256,3),np.uint8)
    for label in range(21):
        code=label
        for j in range(8):
            for c in range(3):tab[label,c]|=((code>>c)&1)<<(7-j)
            code>>=3
    tab[255]=[255,255,255]
    return tab
PALETTE=voc_palette()


def seg_gallery(path,items,methods,tile=160):
    if not items:return
    from PIL import ImageDraw
    labels=['GT RGB','DeepLab GT']+list(methods)
    margin=10;gap=8;header=50;rowh=22+2*tile+gap+12
    canvas=Image.new('RGB',(margin*2+len(labels)*(tile+gap),header+rowh*len(items)+20),'white')
    draw=ImageDraw.Draw(canvas)
    for ci,label in enumerate(labels):draw.text((margin+ci*(tile+gap),17),label[:23],fill='black')
    for ri,(idx,images,masks) in enumerate(items):
        yy=header+ri*rowh
        draw.text((margin,yy),f'VOC #{idx}',fill='black');yy+=22
        for ci,(img,ma) in enumerate(zip(images,masks)):
            xx=margin+ci*(tile+gap)
            canvas.paste(img.resize((tile,tile),Image.Resampling.BICUBIC),(xx,yy))
            maskimg=Image.fromarray(PALETTE[ma.astype(np.uint8)],'RGB')
            canvas.paste(maskimg.resize((tile,tile),Image.Resampling.NEAREST),(xx,yy+tile+gap))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


@torch.inference_mode()
def do_seg(args,device,adapter,drunet,chosen):
    if args.no_seg:return
    from torchvision.datasets import VOCSegmentation
    from torchvision.models.segmentation import deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
    ds=VOCSegmentation(root=str(args.voc_root),year='2012',image_set='val',download=False)
    weights=DeepLabV3_ResNet50_Weights.DEFAULT
    segnet=deeplabv3_resnet50(weights=weights).to(device).eval().requires_grad_(False)
    transform=weights.transforms()
    def predict(srgb,shape):
        inp=transform(srgb.float().clamp(0,1)[0]).unsqueeze(0)
        logits=segnet(inp)['out'].float()
        return F.interpolate(logits,size=shape,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)
    def addcm(cm,pred,truth):
        good=(truth!=255)&(truth<21)&(pred<21)
        inds=truth[good].astype(int)*21+pred[good].astype(int)
        cm+=np.bincount(inds,minlength=441).reshape(21,21)
    def scores(cm,present):
        tp=np.diag(cm).astype(float);den=cm.sum(0)+cm.sum(1)-tp
        keep=present & (den>0);fg=keep.copy();fg[0]=False
        iou=tp/np.maximum(den,1)
        return {'miou_gt_present':float(iou[keep].mean()) if keep.any() else float('nan'),
                'foreground_miou':float(iou[fg].mean()) if fg.any() else float('nan'),
                'pixel_accuracy':float(tp.sum()/max(cm.sum(),1))}
    methods=['gt_rgb','v15_full','adapter_ddcolor','drunet_direct_ddcolor','pnp_ddcolor']
    v15full=v16.load_v15(args,device,'full');v20edge=optional_v20_edge(args,device)
    if v20edge is not None:methods.append('v20_edge_ddcolor')
    dd=load_colorizer(args,device)
    cms={(B,m):np.zeros((21,21),np.int64) for B in args.budgets for m in methods}
    gl={B:[] for B in args.budgets};per=[]
    begin=args.skip_voc;n=min(args.seg_images,len(ds)-begin)
    if n<=0:raise ValueError('Insufficient VOC samples after --skip-voc')
    for i in range(n):
        index=begin+i
        image,mask=ds[index];w,h=image.size;s=min(w,h)
        r=((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2)
        crop=image.convert('RGB').crop(r).resize((args.size,args.size),Image.Resampling.BICUBIC)
        mask=mask.crop(r).resize((args.size,args.size),Image.Resampling.NEAREST)
        truth=np.asarray(mask,dtype=np.uint8)
        srgb=torch.from_numpy(np.asarray(crop,dtype=np.float32).copy()).permute(2,0,1)[None].to(device)/255
        gt=v15.srgb_to_linear(srgb)
        predgt=predict(srgb,truth.shape)
        for B in args.budgets:
            youts,k,_=image_measures(gt,index,B,args,adapter,drunet,chosen[str(B)])
            v15.seed_photons(args,index,B,salt=300000)
            feats,*_=v15.make_features(gt,B,args)
            full=v15full(v16.make_variant(feats,'full'))
            if v20edge is not None:youts['v20_edge']=v20edge(feats,youts['adapter']).clamp(0,1)
            rgbs={'gt_rgb':gt,'v15_full':full}
            rgbs.update({m+'_ddcolor':v16.ddcolorize(dd,y,device) for m,y in youts.items()})
            pred={'gt_rgb':predgt}
            for m in methods[1:]:pred[m]=predict(v15.linear_to_srgb(rgbs[m]),truth.shape)
            present=np.bincount(truth[truth<21],minlength=21)>0
            for m in methods:
                cm=np.zeros((21,21),np.int64);addcm(cm,pred[m],truth);addcm(cms[(B,m)],pred[m],truth)
                per.append({'voc_index':index,'budget_total':B,'method':m,**scores(cm,present)})
            if len(gl[B])<args.gallery_n:
                imgs=[crop,crop]+[v16.to_pil_srgb(rgbs[m]) for m in methods[1:]]
                masks=[truth,predgt]+[pred[m] for m in methods[1:]]
                gl[B].append((index,imgs,masks))
        if (i+1)%5==0 or i+1==n:print('VOC',i+1,'/',n,flush=True)
    all_present=cms[(args.budgets[0],'gt_rgb')].sum(1)>0
    summary=[]
    for B in args.budgets:
        for m in methods:summary.append({'budget_total':B,'method':m,'images':n,**scores(cms[(B,m)],all_present)})
        seg_gallery(args.out/'voc_seg'/f'seg_gallery_B{B:g}.png',gl[B],methods[1:])
    write_csv(args.out/'voc_seg'/'summary.csv',summary)
    write_csv(args.out/'voc_seg'/'per_image.csv',per)
    print('Saved VOC:',args.out/'voc_seg',flush=True)



def make_decision_report(args):
    def read(path):
        if not path.is_file():return []
        with path.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    val=read(args.out/'validation'/'summary.csv')
    lum=read(args.out/'imagenette_test'/'luma_summary.csv')
    rgb=read(args.out/'imagenette_test'/'rgb_summary.csv')
    seg=read(args.out/'voc_seg'/'summary.csv')
    def pick(rows,B,method,key):
        vals=[float(r[key]) for r in rows if abs(float(r['budget_total'])-B)<1e-8 and r['method']==method]
        return vals[0] if vals else float('nan')
    lines=['SPAD DRUNet / exact Bernoulli PnP - automatic decision report',
           'This is a small screening benchmark, NOT final statistical proof.',
           f'Val scenes: {args.val_images}; independent Imagenette test scenes: {args.test_images}; VOC after skip {args.skip_voc}: {args.seg_images}',
           'All comparisons use matched photon observations and pretrained frozen priors.',
           'Measurement NLL is in-sample (not heldout), so never use it alone as proof of better recovery.','']
    if val:
        lines.append('Validation-selected PnP hyperparameters (see selected.json):')
        lines.append((args.out/'selected.json').read_text(encoding='utf-8'))
    for B in args.budgets:
        lines.append(f'B={B:g}')
        l0=pick(lum,B,'adapter','luma_mae');l1=pick(lum,B,'pnp','luma_mae')
        e0=pick(lum,B,'adapter','edge_mae');e1=pick(lum,B,'pnp','edge_mae')
        pfull=pick(rgb,B,'v15_full','psnr_srgb');ppnp=pick(rgb,B,'pnp_ddcolor','psnr_srgb')
        lines.append(f'  Y MAE adapter {l0:.6f}; PnP {l1:.6f}; delta PnP-adapter {l1-l0:+.6f} (lower better)')
        lines.append(f'  Edge MAE adapter {e0:.6f}; PnP {e1:.6f}; delta {e1-e0:+.6f} (lower better)')
        lines.append(f'  RGB PSNR V15 Full {pfull:.3f}; PnP+DDColor {ppnp:.3f}; delta {ppnp-pfull:+.3f}dB')
        if seg:
            va=pick(seg,B,'adapter_ddcolor','miou_gt_present');vp=pick(seg,B,'pnp_ddcolor','miou_gt_present')
            lines.append(f'  VOC mIoU Adapter+DDColor {va:.3%}; PnP+DDColor {vp:.3%}; delta {(vp-va)*100:+.2f} percentage points')
        if math.isfinite(l0) and math.isfinite(l1) and math.isfinite(e0) and math.isfinite(e1):
            if l1<l0 and e1<e0:lines.append('  Screen: PnP improved both Y MAE and edge MAE on this test subset.')
            else:lines.append('  Screen: no simultaneous improvement in both Y MAE and edge MAE.')
        lines.append('')
    lines+=['Interpretation:',
            'Advance only with robust independent-scene gains in luminance, RGB, and/or semantic performance.',
            'If no consistent improvement at B=8, stop expensive PnP expansion and examine sensor information limits.',
            'A color prior does not make monochrome SPAD measurement spectrally identifiable.']
    out=args.out/'PNP_DECISION_REPORT.txt';out.write_text('\n'.join(lines),encoding='utf-8')
    print('Decision report:',out,flush=True)

def selftest():
    # Verify implementation directly against the previously SciPy-verified NumPy proximal.
    from SPAD_binomial_prox import prox_binomial
    r=np.random.default_rng(101)
    for b in [.5,2,8]:
        y=r.normal(.4,.3,size=(1,1,12,12)).astype(np.float32)
        k=r.integers(0,65,size=(1,1,12,12)).astype(np.float32)
        for rho in [12.,40.]:
            val=exact_binomial_prox_torch(torch.from_numpy(y),torch.from_numpy(k),b,64,.01,rho).numpy()
            want=prox_binomial(y,k,b,rho,64,.01)
            err=np.max(np.abs(val-want))
            if err>5e-6: raise AssertionError(f'Prox mismatch B={b} rho={rho} err={err}')
    # Invariance check: GT is passed only to make_features to simulate K; inference uses K/feats.
    print('PASS: PyTorch Bernoulli proximal agrees with independently verified NumPy solver.')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','all'],nargs='?',default='all')
    p.add_argument('--data',type=Path,default=Path('data/imagenette2-160'))
    p.add_argument('--voc-root',type=Path,default=Path('data/VOC'))
    p.add_argument('--v15-out',type=Path,default=Path('runs/v15_64'))
    p.add_argument('--v17-out',type=Path,default=Path('runs/v17_adapter'))
    p.add_argument('--v20-out',type=Path,default=Path('runs/v20_structural'))
    p.add_argument('--out',type=Path,default=Path('runs/spad_dpir_pnp'))
    p.add_argument('--device',choices=['cuda','cpu','auto'],default='auto')
    p.add_argument('--dpir-root',type=Path,default=Path('external/DPIR'))
    p.add_argument('--weights',type=Path,default=None)
    p.add_argument('--ddcolor-repo',type=Path,default=Path('external/DDColor'))
    p.add_argument('--ddcolor-model',default='piddnad/ddcolor_paper_tiny')
    p.add_argument('--ddcolor-weights',type=Path,default=None)
    p.add_argument('--ddcolor-model-size',choices=['tiny','large'],default='tiny')
    p.add_argument('--ddcolor-size',type=int,default=256)
    p.add_argument('--size',type=int,default=128);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--val-count',type=int,default=500)
    p.add_argument('--mode',choices=['bernoulli'],default='bernoulli')
    p.add_argument('--cycles',type=int,default=64);p.add_argument('--bg',type=float,default=.01)
    p.add_argument('--budgets',type=float,nargs='+',default=[.5,2.,8.])
    p.add_argument('--bf-radius',type=int,default=3);p.add_argument('--bf-sigma',type=float,default=1.5)
    p.add_argument('--bf-knoise',type=float,default=2.);p.add_argument('--bf-prec-cap',type=float,default=4.)
    p.add_argument('--val-images',type=int,default=10)
    p.add_argument('--test-images',type=int,default=20)
    p.add_argument('--seg-images',type=int,default=12)
    p.add_argument('--skip-voc',type=int,default=50)
    p.add_argument('--gallery-n',type=int,default=5)
    p.add_argument('--direct-sigma',type=float,default=15./255.)
    p.add_argument('--no-seg',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if args.size%8:raise ValueError('Size must be divisible by 8')
    if args.cycles!=64:raise ValueError('This benchmark locks 64 exposures for existing V17 compatibility')
    args.out.mkdir(parents=True,exist_ok=True)
    selftest()
    if args.action=='check':return
    device=device_of(args)
    print('Device:',device,'No neural network training in this experiment',flush=True)
    adapter,drunet=make_models(args,device)
    chosen=do_val(args,device,adapter,drunet)
    do_test(args,device,adapter,drunet,chosen)
    if not args.no_seg:do_seg(args,device,adapter,drunet,chosen)
    make_decision_report(args)
    (args.out/'run_config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2),encoding='utf-8')
    print('DONE:',args.out,flush=True)

if __name__=='__main__':
    main()
