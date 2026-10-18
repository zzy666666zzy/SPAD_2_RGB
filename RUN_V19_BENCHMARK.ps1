# Run from any folder: this file and v19_seg_benchmark_fixed.py must be in D:\low_count_to_rgb.
# No checkpoint is changed or overwritten by this benchmark.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$required = @(
    'train_eval_v19.py',
    'train_eval_v18.py',
    'eval_v18_segmentation.py',
    'runs\v18_bridge_fast\bridge_best.pt',
    'runs\v19_seg\A\bridge_last.pt',
    'runs\v19_seg\B\bridge_last.pt'
)
foreach ($file in $required) {
    if (!(Test-Path $file)) {
        Write-Host "MISSING: $file" -ForegroundColor Red
        throw "Please check your V18/V19 checkpoint path above. Nothing was modified."
    }
}

Write-Host 'STAGE 1/2: Evaluate frozen DeepLab on GT RGB (native vs crop vs old preprocessing).' -ForegroundColor Cyan
python v19_seg_benchmark_fixed.py sanity --voc-root data/VOC --out runs/v19_seg/benchmark_fixed --device cuda --segmentor deeplab_resnet50 --size 128 --skip-images 6 --eval-images 20 --gallery-n 6 --tile 160
if ($LASTEXITCODE -ne 0) { throw 'STAGE 1 failed; STAGE 2 was NOT started.' }

Write-Host 'STAGE 2/2: Compare V15/V18/V19 on IDENTICAL photon observations and official preprocessing.' -ForegroundColor Cyan
python v19_seg_benchmark_fixed.py compare --voc-root data/VOC --v15-out runs/v15_64 --v17-out runs/v17_adapter --v18-out runs/v18_bridge_fast --v19-out runs/v19_seg --out runs/v19_seg/benchmark_fixed --device cuda --segmentor deeplab_resnet50 --size 128 --mode bernoulli --cycles 64 --bg 0.01 --budgets 0.5 2 8 --skip-images 6 --eval-images 20 --gallery-n 6 --tile 160 --steps 4 --checkpoint last --sample-noise 0 --methods raw full bridge v19_a v19_b --native-reference
if ($LASTEXITCODE -ne 0) { throw 'STAGE 2 failed; original checkpoints were NOT modified.' }

Write-Host 'SUCCESS. Results: runs\v19_seg\benchmark_fixed' -ForegroundColor Green
Write-Host 'Send me: sanity_summary.csv, compare_summary.csv, seg_gallery_B2.png' -ForegroundColor Green
