$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
Write-Host 'V20 one-shot experiment: 3 paired luminance models + RGB + official VOC segmentation' -ForegroundColor Cyan
$required = @('train_eval_v20.py','train_eval_v17.py','v17_v15_compat.py','v17_v16_compat.py',
'data/imagenette2-160/train','data/imagenette2-160/val','data/VOC/VOCdevkit/VOC2012',
'runs/v17_adapter/luma_adapter_best.pt','runs/v15_64/full_best.pt')
$missing = @($required | Where-Object { -not (Test-Path $_) })
if ($missing.Count -gt 0) {
    Write-Host ('Missing required paths: ' + ($missing -join ', ')) -ForegroundColor Red
    throw 'Cannot start V20 until required local project files are present.'
}
$cmd = @('train_eval_v20.py', 'all',
    '--data', 'data/imagenette2-160',
    '--voc-root', 'data/VOC',
    '--v15-out', 'runs/v15_64',
    '--v17-out', 'runs/v17_adapter',
    '--out', 'runs/v20_structural',
    '--device', 'cuda',
    '--mode', 'bernoulli', '--cycles','64','--bg','0.01',
    '--budgets','0.5','1','2','4','8',
    '--seg-budgets','0.5','2','8',
    '--size','128','--base','24',
    '--epochs','3','--train-images','3000','--batch','8','--workers','2',
    '--val-images','24','--patience','2','--amp','fp16',
    '--eval-images','30','--seg-eval-images','20','--skip-voc','6',
    '--gallery-n','6','--tensorboard','--tb-train-images','2',
    '--tb-val-images','3','--tb-budgets','0.5','2','8',
    '--tb-train-every','300','--log-every','40',
    '--ddcolor-repo','external/DDColor',
    '--segmentor','deeplab_resnet50')
& python @cmd
if ($LASTEXITCODE -ne 0) { throw "V20 exited with code $LASTEXITCODE. Previous checkpoints are unchanged." }
Write-Host 'V20 finished successfully! Open runs/v20_structural/V20_DECISION_REPORT.txt' -ForegroundColor Green
