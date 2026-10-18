#!/usr/bin/env python3
"""Multi-scene real-data comparison against the dataset's own reconstructions (bit2bit).

For every scene folder under --root that has inference.tif (+ input.tif, + a zip of .bin):
  1. import the zip into <out>/<scene-slug>_<N>f/ (N = frames kept)                [spad_import]
  2. verify that input.tif (the 512 frames bit2bit reconstructed) == zip frames at some offset
  3. reconstruct OURS at several windows T for the displayed/scored frames only       [spad_physics_v13]
  4. pseudo-GT = MLE of the last --pseudo-gt frames (disjoint), scored inside --roi per scene
Outputs in --out:
  grid.gif          rows = scenes, columns = --gif-cols (default raw, ours:<W>, bit2bit, pseudogt)
  frame_grid.png    rows = scenes, columns = ours at --grid-windows for frame --frame (+ optional cols)
  scene_summary.csv ROI PSNR vs pseudo-GT per scene / method / T
  <slug>_<N>f/      imported frames, meta.json (with alignment offset, roi)

Example
  python run_scenes.py --root data/bit2bit --out runs/scenes --max-frames 60000 --pseudo-gt 50000 \\
      --gif-cols raw ours:64 bit2bit pseudogt --grid-windows 50 100 200 500 --frame 256 \\
      --roi-file rois.json --backend torch
rois.json: {"cpu_fan_with_resolution_target": [150,350,150,350], "cpu_fan_with_mandrill_image": [100,400,100,400]}
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spad_physics_v13 as sp  # noqa: E402

IMPORT = HERE / 'spad_import.py'


def slug(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower()


def find_sources(folder, inference_name, input_name):
    inf = folder / inference_name
    inp = folder / input_name
    zips = []
    import zipfile
    for z in sorted(folder.glob('*.zip')):
        try:
            with zipfile.ZipFile(z) as zz:
                if any(n.lower().endswith('.bin') for n in zz.namelist()):
                    zips.append(z)
        except Exception:
            pass
    return (inf if inf.is_file() else None), (inp if inp.is_file() else None), zips


def frame_hashes(frames):
    return [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() for f in frames]


def align_input(frames, input_tif, max_search):
    """Return offset k such that input.tif[i] == frames[k+i] for all 512 frames, else None."""
    import tifffile
    inp = (tifffile.imread(input_tif) > 0).astype(np.uint8)
    if inp.shape[1:] != frames.shape[1:]:
        return None, inp
    h0 = frame_hashes(inp[:1])[0]
    hs = frame_hashes(frames[:min(max_search, len(frames))])
    for k, h in enumerate(hs):
        if h == h0 and k + len(inp) <= len(frames) and np.array_equal(frames[k:k + len(inp)], inp):
            return k, inp
    return None, inp


def window_recon(frames, alpha, bg, hot, T, i, m):
    n = frames.shape[0]
    a, b = max(0, i - T // 2), min(n, i - T // 2 + T)
    N = frames[a:b].sum(0, dtype=np.float64); Tw = b - a
    p = np.clip((N + m) / (Tw + 1.), 1e-6, 1 - 1e-6)
    rho = np.clip((-np.log1p(-p) - bg) / np.maximum(alpha, 1e-9), 0, None)
    var = 1. / np.maximum(Tw * np.square(alpha) * (1 - p) / p, 1e-9)
    rho = sp._fill_hot(rho, hot)
    return sp.precision_bilateral(rho, var, 3, 1.5), N / Tw


def to_disp(x, lo, hi, gamma=True):
    g = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
    return sp.to_img(sp.linear_to_srgb(g) if gamma else g)


def grid_image(rows, col_titles, cell, row_titles=None):
    left = 120 if row_titles else 0; header = 30
    im = Image.new('RGB', (left + cell * len(col_titles), header + (cell + 4) * len(rows)), 'white')
    d = ImageDraw.Draw(im)
    for j, t in enumerate(col_titles):
        d.multiline_text((left + cell * j + 4, 4), t, fill='black')
    for r, imgs in enumerate(rows):
        y = header + r * (cell + 4)
        if row_titles:
            d.multiline_text((4, y + 6), row_titles[r], fill='black')
        for j, x in enumerate(imgs):
            im.paste(x.resize((cell, cell), Image.Resampling.LANCZOS), (left + cell * j, y))
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scenes', action='append', default=[], help='substring filter (repeatable)')
    ap.add_argument('--inference-name', default='inference.tif')
    ap.add_argument('--input-name', default='input.tif')
    ap.add_argument('--max-frames', type=int, default=60000, help='frames to import from the zip')
    ap.add_argument('--pseudo-gt', type=int, default=50000, help='last N imported frames form the reference')
    ap.add_argument('--crop', type=int, nargs=4, default=None, metavar=('Y0', 'Y1', 'X0', 'X1'))
    ap.add_argument('--alpha', type=float, default=1.)
    ap.add_argument('--roi-file', type=Path, default=None, help='json {scene_slug: [y0,y1,x0,x1]}')
    ap.add_argument('--gif-cols', nargs='+', default=['raw', 'ours:64', 'bit2bit', 'pseudogt'],
                    help='raw | bits:<T> | ours:<T> | bit2bit | pseudogt')
    ap.add_argument('--gif-frames', type=int, default=32, help='frames in the gif (evenly spaced over the 512)')
    ap.add_argument('--gif-ms', type=int, default=120)
    ap.add_argument('--grid-windows', type=int, nargs='+', default=[50, 100, 200, 500])
    ap.add_argument('--grid-extra', nargs='*', default=['bit2bit'], help='extra columns in frame_grid: raw | bit2bit | pseudogt')
    ap.add_argument('--frame', type=int, default=256, help='frame index (within input.tif) for frame_grid.png')
    ap.add_argument('--cell', type=int, default=220)
    ap.add_argument('--backend', choices=['auto', 'numpy', 'torch'], default='numpy')
    ap.add_argument('--skip-import', action='store_true', help='reuse existing <slug>_<N>f folders')
    args = ap.parse_args()
    print('backend:', sp.set_backend(args.backend), flush=True)

    rois = json.loads(args.roi_file.read_text()) if args.roi_file else {}
    folders = [p for p in sorted(args.root.iterdir()) if p.is_dir()]
    if args.scenes:
        folders = [p for p in folders if any(s.lower() in p.name.lower() for s in args.scenes)]
    args.out.mkdir(parents=True, exist_ok=True)
    gif_rows, grid_rows, row_titles, summary = [], [], [], []
    gif_frames_idx = None
    for folder in folders:
        inf, inp, zips = find_sources(folder, args.inference_name, args.input_name)
        if inf is None or inp is None or not zips:
            print('skip %s (needs %s, %s and a zip with .bin)' % (folder.name, args.inference_name, args.input_name)); continue
        sl = slug(folder.name)
        dst = args.out / ('%s_%df' % (sl, args.max_frames))
        if not (args.skip_import and (dst / 'frames.npy').exists()):
            cmd = [sys.executable, str(IMPORT), '--src', str(zips[0]), '--format', 'zip', '--out', str(dst), '--alpha', str(args.alpha), '--max-frames', str(args.max_frames)]
            if args.crop: cmd += ['--crop', *map(str, args.crop)]
            print('>>', ' '.join(cmd), flush=True)
            if subprocess.run(cmd).returncode:
                print('!! import failed for', folder.name); continue
        meta = json.loads((dst / 'meta.json').read_text())
        frames = np.load(dst / 'frames.npy', mmap_mode='r')
        alpha = float(meta.get('alpha', 1.)); bg = np.load(dst / meta['bg_map']) if meta.get('bg_map') else 0.
        hot = np.load(dst / meta['hot_mask']) if meta.get('hot_mask') else None
        if args.crop and bg is not None and np.ndim(bg) == 2 and bg.shape != frames.shape[1:]:
            pass
        # ---- alignment of input.tif with the zip
        off, inp_frames = align_input(np.asarray(frames[:4096]), inp, 4096)
        if off is None:
            print('!! %s: input.tif not found in the first 4096 zip frames; using input.tif frames directly, no pseudo-GT alignment guarantee' % folder.name)
            eval_frames = inp_frames; eval_off = None
        else:
            print('%s: input.tif == zip frames [%d, %d)' % (folder.name, off, off + len(inp_frames)), flush=True)
            eval_frames = np.asarray(frames[off:off + len(inp_frames)]); eval_off = off
        meta.update(input_tif_offset=eval_off, inference=str(inf.resolve()), roi=rois.get(sl)); (dst / 'meta.json').write_text(json.dumps(meta, indent=2))
        import tifffile
        b2b = tifffile.imread(inf).astype(np.float32)
        n_eval = len(eval_frames)
        # ---- pseudo-GT (disjoint)
        ref = None
        if args.pseudo_gt > 0:
            lo_i = frames.shape[0] - args.pseudo_gt
            if eval_off is not None and lo_i < eval_off + n_eval:
                print('!! pseudo-GT overlaps evaluated frames; reduce --pseudo-gt'); lo_i = max(lo_i, eval_off + n_eval)
            ref, _, _ = sp.bernoulli_mle(np.asarray(frames[lo_i:]), alpha, bg)
            ref = sp._fill_hot(ref, hot)
        roi = rois.get(sl, [0, frames.shape[1], 0, frames.shape[2]]); y0, y1, x0, x1 = roi
        m = float(eval_frames.mean())
        # ---- which frames do we need
        if gif_frames_idx is None:
            gif_frames_idx = list(np.linspace(0, n_eval - 1, args.gif_frames).astype(int))
        need_T = sorted({int(c.split(':')[1]) for c in args.gif_cols if ':' in c} | set(args.grid_windows))
        cache = {}  # (T, i) -> (ours, bits_mean)
        for T in need_T:
            idx = set(gif_frames_idx if any(c == 'ours:%d' % T or c == 'bits:%d' % T for c in args.gif_cols) else []) | ({args.frame} if T in args.grid_windows else set())
            for i in sorted(idx):
                cache[(T, i)] = window_recon(eval_frames, alpha, bg, hot, T, i, m)
            print('  %s: T=%d reconstructed %d frames' % (sl, T, len(idx)), flush=True)
        # ---- scoring in ROI vs pseudo-GT (global gain per method)
        def score(x):
            if ref is None: return float('nan')
            R = ref[y0:y1, x0:x1]; rs = float(np.percentile(R, 99.5)); Rn = np.clip(R / rs, 0, 1)
            X = x[y0:y1, x0:x1].astype(np.float64); g = float(Rn.mean() / max(X.mean(), 1e-12))
            return sp.psnr(sp.linear_to_srgb(np.clip(X * g, 0, 1)), sp.linear_to_srgb(Rn))
        for T in args.grid_windows:
            o, mb = cache[(T, args.frame)]
            summary += [{'scene': sl, 'frames_imported': int(frames.shape[0]), 'T': T, 'frame': args.frame, 'method': 'ours', 'roi_psnr': score(o)},
                        {'scene': sl, 'frames_imported': int(frames.shape[0]), 'T': T, 'frame': args.frame, 'method': 'mean-of-bits', 'roi_psnr': score(mb)}]
        summary.append({'scene': sl, 'frames_imported': int(frames.shape[0]), 'T': 'n/a', 'frame': args.frame, 'method': 'bit2bit', 'roi_psnr': score(b2b[args.frame])})
        # ---- display scales: per column type, from the data itself
        ref_lo, ref_hi = (np.percentile(ref, .5), np.percentile(ref, 99.5)) if ref is not None else (0, 1)
        b_lo, b_hi = np.percentile(b2b.mean(0), .5), np.percentile(b2b.mean(0), 99.5)
        def ours_scale(T):
            arr = np.stack([cache[(T, i)][0] for i in gif_frames_idx if (T, i) in cache] or [cache[(T, args.frame)][0]])
            return np.percentile(arr.mean(0), .5), np.percentile(arr.mean(0), 99.5)
        # gif row: list over gif frames of list over columns
        row = []
        for i in gif_frames_idx:
            cols = []
            for c in args.gif_cols:
                if c == 'raw': cols.append(sp.to_img(eval_frames[i]))
                elif c == 'bit2bit': cols.append(to_disp(b2b[i], b_lo, b_hi))
                elif c == 'pseudogt': cols.append(to_disp(ref, ref_lo, ref_hi) if ref is not None else sp.to_img(np.zeros_like(eval_frames[0], float)))
                else:
                    kind, T = c.split(':'); T = int(T); o, mb = cache[(T, i)]
                    lo, hi = ours_scale(T) if kind == 'ours' else (0, float(np.percentile(mb, 99.5)))
                    cols.append(to_disp(o if kind == 'ours' else mb, lo, hi, gamma=(kind == 'ours')))
            row.append(cols)
        gif_rows.append(row)
        # frame grid row
        g = []
        for T in args.grid_windows:
            lo, hi = ours_scale(T); g.append(to_disp(cache[(T, args.frame)][0], lo, hi))
        for c in args.grid_extra:
            if c == 'raw': g.append(sp.to_img(eval_frames[args.frame]))
            elif c == 'bit2bit': g.append(to_disp(b2b[args.frame], b_lo, b_hi))
            elif c == 'pseudogt' and ref is not None: g.append(to_disp(ref, ref_lo, ref_hi))
        grid_rows.append(g)
        row_titles.append(folder.name[:18] + '\n%d frames' % frames.shape[0])
    if not gif_rows:
        sys.exit('nothing processed')
    sp.write_csv(args.out / 'scene_summary.csv', summary)
    # ---- frame grid png with ROI PSNR in titles
    titles = ['ours T=%d' % T for T in args.grid_windows] + [c for c in args.grid_extra]
    rows_png = []
    for r, g in enumerate(grid_rows):
        rows_png.append(g)
    im = grid_image(rows_png, titles, args.cell, row_titles)
    d = ImageDraw.Draw(im)
    for r, sl_ in enumerate([s['scene'] for s in summary[::1] if s['method'] == 'bit2bit']):
        for j, T in enumerate(args.grid_windows):
            v = [s['roi_psnr'] for s in summary if s['scene'] == sl_ and s['method'] == 'ours' and s['T'] == T]
            if v and v[0] == v[0]:
                d.text((120 + args.cell * j + 4, 30 + r * (args.cell + 4) + 4), '%.1f dB' % v[0], fill='yellow')
        v = [s['roi_psnr'] for s in summary if s['scene'] == sl_ and s['method'] == 'bit2bit']
        if 'bit2bit' in args.grid_extra and v and v[0] == v[0]:
            j = len(args.grid_windows) + args.grid_extra.index('bit2bit')
            d.text((120 + args.cell * j + 4, 30 + r * (args.cell + 4) + 4), '%.1f dB' % v[0], fill='yellow')
    im.save(args.out / 'frame_grid.png')
    # ---- gif
    frames_out = []
    for k, i in enumerate(gif_frames_idx):
        frames_out.append(grid_image([row[k] for row in gif_rows], [c + ('  f=%d' % i if j == 0 else '') for j, c in enumerate(args.gif_cols)], args.cell, row_titles))
    frames_out[0].save(args.out / 'grid.gif', save_all=True, append_images=frames_out[1:], duration=args.gif_ms, loop=0)
    print('\nROI PSNR vs pseudo-GT (frame %d):' % args.frame)
    for s in summary:
        print('  %-36s %-12s T=%-5s %s' % (s['scene'], s['method'], s['T'], ('%.2f dB' % s['roi_psnr']) if s['roi_psnr'] == s['roi_psnr'] else 'n/a'))
    print('Saved', args.out / 'grid.gif', args.out / 'frame_grid.png', args.out / 'scene_summary.csv')


if __name__ == '__main__':
    main()
