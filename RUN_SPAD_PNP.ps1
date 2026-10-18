# Execute from project root: powershell -NoProfile -ExecutionPolicy Bypass -File .\RUN_SPAD_PNP.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
Write-Host '=== SPAD PnP: verify physical math ===' -ForegroundColor Cyan
python run_spad_pnp.py check
if ($LASTEXITCODE -ne 0) { throw 'Bernoulli solver check failed; no evaluation was started.' }
Write-Host '=== SPAD PnP: validation selection -> independent RGB test -> VOC segmentation ===' -ForegroundColor Cyan
python run_spad_pnp.py all --data data/imagenette2-160 --voc-root data/VOC --v15-out runs/v15_64 --v17-out runs/v17_adapter --v20-out runs/v20_structural --out runs/spad_dpir_pnp --device cuda --mode bernoulli --cycles 64 --bg 0.01 --size 128 --budgets 0.5 2 8 --val-images 10 --test-images 20 --seg-images 12 --skip-voc 50 --gallery-n 5 --ddcolor-repo external/DDColor
if ($LASTEXITCODE -ne 0) { throw 'PnP evaluation stopped. Send me the full error text; do not retrain V15-V20.' }
Write-Host '=== DONE: send runs/spad_dpir_pnp/PNP_DECISION_REPORT.txt and the summary CSVs ===' -ForegroundColor Green
