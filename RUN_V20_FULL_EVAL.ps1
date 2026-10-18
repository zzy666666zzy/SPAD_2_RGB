$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$common=@('--data','data/imagenette2-160','--voc-root','data/VOC',
'--v15-out','runs/v15_64','--v17-out','runs/v17_adapter',
'--out','runs/v20_structural','--device','cuda','--mode','bernoulli',
'--cycles','64','--bg','0.01','--budgets','0.5','1','2','4','8','--size','128')
& python train_eval_v20.py eval_rgb @common --split test --eval-images 0 --gallery-n 6
if ($LASTEXITCODE -ne 0) {throw 'RGB evaluation failed'}
& python train_eval_v20.py eval_seg @common --seg-budgets 0.5 1 2 4 8 --seg-eval-images 0 --skip-voc 6 --gallery-n 6 --segmentor deeplab_resnet50
if ($LASTEXITCODE -ne 0) {throw 'VOC evaluation failed'}
& python train_eval_v20.py report @common --seg-budgets 0.5 1 2 4 8
if ($LASTEXITCODE -ne 0) {throw 'Report generation failed'}
Write-Host 'Full evaluation completed.' -ForegroundColor Green
