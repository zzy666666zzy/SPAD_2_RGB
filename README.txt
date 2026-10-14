SPAD Mac local benchmark v2

Use your existing Python/Conda environment.

Install dependencies (if missing):
python -m pip install torch numpy pillow scikit-image

Train:
python spad_mac_quickbench.py --train-count 800 --val-count 100 --test-count 100 --size 128 --batch 8 --steps 600 --ddpm-steps 800 --device auto --out runs/spad

Inference:
python spad_mac_quickbench.py \
  --infer-input spad_intensity.npy \
  --infer-method best \
  --infer-photons 10 \
  --device auto \
  --out runs/spad

Includes supervised direct and residual U-Nets plus tiny conditional DDPM. External pretrained DDColor/gQIR weights are NOT included.
