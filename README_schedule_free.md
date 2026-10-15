# REPA Training Runs

Run each command separately from this repository in a Bash shell. Replace the
two dataset paths with your training and validation data paths. The VAE uses
upstream REPA's `stabilityai/sd-vae-ft-mse` model. Each run uses four GPUs,
global batch size 128, seed 0,
and up to 800k training steps. Fixed evaluation runs every 2,000 steps with
the same output directory and noise-cache settings across all three runs.

## 1. AdamW Baseline: LR 1e-4, No Warmup

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 train.py \
  --report-to=wandb \
  --allow-tf32 \
  --mixed-precision=bf16 \
  --seed=0 \
  --path-type=linear \
  --prediction=v \
  --weighting=uniform \
  --model=SiT-XL/2 \
  --resolution=256 \
  --enc-type=dinov2-vit-b \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir=exps/schedule_free \
  --exp-name=adamw-lr1e-4 \
  --vae-path=stabilityai/sd-vae-ft-mse \
  --data-dir=/path/to/imagenet_train_features \
  --eval-data-dir=/path/to/imagenet_val_features \
  --batch-size=128 \
  --gradient-accumulation-steps=1 \
  --optimizer-mode=adamw \
  --learning-rate=1e-4 \
  --lr-schedule=constant \
  --warmup-steps=0 \
  --adam-beta1=0.9 \
  --adam-beta2=0.999 \
  --adam-epsilon=1e-8 \
  --adam-weight-decay=0 \
  --max-grad-norm=1.0 \
  --max-train-steps=800000 \
  --checkpointing-steps=20000 \
  --sampling-steps=20000 \
  --fixed-eval-steps=2000 \
  --fixed-eval-num-samples=10000 \
  --fixed-eval-mode=t \
  --fixed-eval-ts=0.02,0.1,0.2,0.4,0.6,0.8,0.9,0.96 \
  --fixed-eval-include-random \
  --fixed-eval-seed=1234
```

## 2. AdamW: LR 4e-4, 5k Linear Warmup

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 train.py \
  --report-to=wandb \
  --allow-tf32 \
  --mixed-precision=bf16 \
  --seed=0 \
  --path-type=linear \
  --prediction=v \
  --weighting=uniform \
  --model=SiT-XL/2 \
  --resolution=256 \
  --enc-type=dinov2-vit-b \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir=exps/schedule_free \
  --exp-name=adamw-lr4e-4-warmup5k \
  --vae-path=stabilityai/sd-vae-ft-mse \
  --data-dir=/path/to/imagenet_train_features \
  --eval-data-dir=/path/to/imagenet_val_features \
  --batch-size=128 \
  --gradient-accumulation-steps=1 \
  --optimizer-mode=adamw \
  --learning-rate=4e-4 \
  --lr-schedule=warmup_constant \
  --warmup-steps=5000 \
  --adam-beta1=0.9 \
  --adam-beta2=0.999 \
  --adam-epsilon=1e-8 \
  --adam-weight-decay=0 \
  --max-grad-norm=1.0 \
  --max-train-steps=800000 \
  --checkpointing-steps=20000 \
  --sampling-steps=20000 \
  --fixed-eval-steps=2000 \
  --fixed-eval-num-samples=10000 \
  --fixed-eval-mode=t \
  --fixed-eval-ts=0.02,0.1,0.2,0.4,0.6,0.8,0.9,0.96 \
  --fixed-eval-include-random \
  --fixed-eval-seed=1234
```

## 3. Schedule-Free AdamW: LR 4e-4, 5k Linear Warmup

Use `--lr-schedule=constant` here: Schedule-Free handles its 5k-step warmup
internally through `--warmup-steps=5000`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 train.py \
  --report-to=wandb \
  --allow-tf32 \
  --mixed-precision=bf16 \
  --seed=0 \
  --path-type=linear \
  --prediction=v \
  --weighting=uniform \
  --model=SiT-XL/2 \
  --resolution=256 \
  --enc-type=dinov2-vit-b \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir=exps/schedule_free \
  --exp-name=sfadamw-lr4e-4-warmup5k \
  --vae-path=stabilityai/sd-vae-ft-mse \
  --data-dir=/path/to/imagenet_train_features \
  --eval-data-dir=/path/to/imagenet_val_features \
  --batch-size=128 \
  --gradient-accumulation-steps=1 \
  --optimizer-mode=adamw_schedulefree \
  --learning-rate=4e-4 \
  --lr-schedule=constant \
  --warmup-steps=5000 \
  --sf-inner-momentum=0 \
  --adam-beta1=0.9 \
  --adam-beta2=0.999 \
  --adam-epsilon=1e-8 \
  --adam-weight-decay=0 \
  --max-grad-norm=1.0 \
  --max-train-steps=800000 \
  --checkpointing-steps=20000 \
  --sampling-steps=20000 \
  --fixed-eval-steps=2000 \
  --fixed-eval-num-samples=10000 \
  --fixed-eval-mode=t \
  --fixed-eval-ts=0.02,0.1,0.2,0.4,0.6,0.8,0.9,0.96 \
  --fixed-eval-include-random \
  --fixed-eval-seed=1234
```
